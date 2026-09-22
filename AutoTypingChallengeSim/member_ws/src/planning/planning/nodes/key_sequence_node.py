"""Node `type_keys`: types the launch key, one aimed and settled press at a time."""
import numpy as np
import tf2_ros
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

from planning.geometry import kinematics as kin
from planning.geometry.approach import approach_point, can_press_from, press_problems
from planning.geometry.keyboard import CELLS, KEYBOARD_ORIGIN, key_center
from planning.geometry.panel import Panel
from planning.utils.ros import LATCHED, WORLD_FRAME, point_msg, run
from planning.utils.tasks import Periodic, TaskError, Tasks, elapsed, hold, sleep

BOARD_FRAME = 'board'

SETTLE_TIME = 0.2       # how long the arm must stay pressable before pressing
SETTLE_TIMEOUT = 8.0    # how long to wait for that before giving up on a key
PRESS_GAP = 0.25        # the simulator ignores a press within 0.15 s of the last accepted one
RESEND_PERIOD = 1.0     # targets are resent until settled, in case `kinematics` was not listening yet
PANEL_SETTLE = 1.0      # when starting by itself, let the board estimate be averaged before committing to it
TF_MAX_AGE = 0.5        # a board transform older than this means its publisher has stopped


class PanelUnavailable(TaskError):
    pass


class KeySequenceNode(Node):

    def __init__(self):
        super().__init__('key_sequence')

        # [x, y] of the key grid's top-left corner in the board frame, in metres.
        self.declare_parameter('keyboard_origin', KEYBOARD_ORIGIN)
        # Start typing on its own once the panel is known, at the first episode and after every reset.
        self.declare_parameter('autostart', False)

        self.arm_angles = None
        self.arm_speed = None
        self.launch_key = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tasks = Tasks(self)

        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.create_subscription(String, '/sim/launch_key', self.launch_key_callback, LATCHED)
        self.head_target_pub = self.create_publisher(PointStamped, '/planning/head_target', 10)
        self.aim_target_pub = self.create_publisher(PointStamped, '/planning/aim_target', 10)
        self.press_pub = self.create_publisher(Empty, '/arm/press', 10)
        self.done_pub = self.create_publisher(Empty, '/sim/done', 10)
        self.create_service(Trigger, '/planning/type_launch_key', self.start_typing)

        self.restart()

    def joint_state_callback(self, msg: JointState):
        positions = dict(zip(msg.name, msg.position))
        velocities = dict(zip(msg.name, msg.velocity))
        try:
            self.arm_angles = np.array([positions[name] for name in kin.JOINT_NAMES])
            self.arm_speed = max(abs(velocities[name]) for name in kin.JOINT_NAMES)
        except KeyError:
            self.get_logger().warn('joint_states missing a joint', throttle_duration_sec=1.0)

    def launch_key_callback(self, msg: String):
        # Only a launch key after the first means a reset, since the first just starts the episode.
        new_episode = self.launch_key is not None
        self.launch_key = msg.data
        if new_episode:
            self.restart()

    def restart(self):
        """Drop whatever was running and, if asked to, wait for the panel and type the launch key."""
        if self.tasks.busy:
            self.get_logger().warn('a new episode started, stopping')
        self.tasks.cancel()
        if self.get_parameter('autostart').value:
            self.tasks.start(self.wait_then_type())

    def start_typing(self, request, response):
        """Service callback to start typing the launch key, one press at a time, waiting for the arm to settle on each key before pressing it."""
        error = self.tasks.start(self.type_launch_key())
        response.success = not error
        response.message = error or f'typing {self.launch_key}'
        return response

    def wait_then_type(self):
        """Wait for the panel to be known, then type the launch key."""
        yield from hold(self.get_clock(), self.not_ready, PANEL_SETTLE)
        yield from self.type_launch_key()

    def not_ready(self):
        """Why typing cannot start yet, or '' when the simulator and the panel are both there."""
        if self.launch_key is None or self.arm_angles is None:
            reason = 'waiting for the simulator'
        else:
            try:
                self.keyboard()
                return ''
            except PanelUnavailable as err:
                reason = f'waiting to start: {err}'
        self.get_logger().info(reason, throttle_duration_sec=5.0)
        return reason

    def type_launch_key(self):
        """Type the launch key, one press at a time, waiting for the arm to settle on each key before pressing it."""
        keys = self.typable_launch_key()
        pressed = 0
        try:
            for key in keys:
                if pressed:
                    yield from sleep(self.get_clock(), PRESS_GAP)
                yield from self.settle_on(key)
                self.press_pub.publish(Empty())
                pressed += 1
                self.get_logger().info(f'pressed {key} ({pressed}/{len(keys)})')
        except TaskError as err:
            raise TaskError(f'stopped after {pressed}/{len(keys)} keys: {err}') from None

        self.get_logger().info(f'typed {keys}')
        self.done_pub.publish(Empty())

    def typable_launch_key(self):
        """Check that the launch key is known, the arm has joint states, and all keys are in the layout."""
        keys = self.launch_key
        if keys is None:
            raise TaskError('no launch key received yet')
        if self.arm_angles is None:
            raise TaskError('no joint_states yet')
        if unknown := sorted(set(keys) - set(CELLS)):
            raise TaskError(f'no layout for {unknown}')
        return keys

    def settle_on(self, key):
        """Aim at a key and wait until the arm is still and pointing at it."""
        clock = self.get_clock()
        resend = Periodic(clock, RESEND_PERIOD, self.aim_at(key))
        yield from hold(clock, lambda: resend() or self.problems(key), SETTLE_TIME, SETTLE_TIMEOUT,
                        what=f'{key} did not settle')

    def aim_at(self, key):
        """Send the targets for a key, a new head position only if pan and tilt alone cannot reach it.

        Returns a function that sends the same targets again.
        """
        panel = self.keyboard()
        target = panel.to_world(key_center(key))
        try:
            head_point = None if can_press_from(self.arm_angles[:3], target, panel) else approach_point(target, panel)
        except kin.IKError as err:
            raise TaskError(f'cannot approach {key}: {err}') from None

        def send():
            if head_point is not None:
                self.head_target_pub.publish(point_msg(head_point, self.get_clock()))
            self.aim_target_pub.publish(point_msg(target, self.get_clock()))

        send()
        move = 'aiming from where the head is' if head_point is None else f'head to {np.round(head_point, 3).tolist()}'
        self.get_logger().info(f'{key} at {np.round(target, 3).tolist()}, {move}')
        return send

    def problems(self, key):
        """What keeps the stylus from pressing `key` right now, or '' when nothing does."""
        hit = self.keyboard().hit(kin.head_position(self.arm_angles), kin.aim_direction(self.arm_angles))
        return '; '.join(press_problems(hit, key_center(key), self.arm_speed))

    def keyboard(self):
        """The key grid as a panel: the board's frame moved to where the grid's top-left corner is."""
        try:
            transform = self.tf_buffer.lookup_transform(WORLD_FRAME, BOARD_FRAME, Time())
        except tf2_ros.TransformException as err:
            raise PanelUnavailable(f'no {BOARD_FRAME} transform: {err}') from None

        age = elapsed(self.get_clock(), Time.from_msg(transform.header.stamp))
        if age > TF_MAX_AGE:
            raise PanelUnavailable(f'{BOARD_FRAME} transform is {age:.1f} s old, its publisher has stopped')

        t, q = transform.transform.translation, transform.transform.rotation
        board = Panel(np.array([t.x, t.y, t.z]), Rotation.from_quat([q.x, q.y, q.z, q.w]))
        return board.shifted(self.get_parameter('keyboard_origin').value)


def main(args=None):
    run(KeySequenceNode, args)


if __name__ == '__main__':
    main()
