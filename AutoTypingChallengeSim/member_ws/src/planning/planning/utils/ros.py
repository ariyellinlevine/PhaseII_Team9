"""ROS plumbing shared by the nodes: joint states, TF frames, the launch key, messages, main()."""
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from planning.geometry import kinematics as kin
from planning.geometry.panel import Panel
from planning.utils.tasks import TaskError, elapsed

WORLD_FRAME = 'world'
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class PanelUnavailable(TaskError):
    pass


def point_msg(point, clock):
    msg = PointStamped()
    msg.header.stamp = clock.now().to_msg()
    msg.header.frame_id = WORLD_FRAME
    msg.point.x, msg.point.y, msg.point.z = point
    return msg


def point_array(msg: PointStamped):
    return np.array([msg.point.x, msg.point.y, msg.point.z])


class ArmState:
    """The latest /joint_states: joint angles in `kin.JOINT_NAMES` order and the fastest joint's speed."""

    def __init__(self, node, on_update=None):
        self.node = node
        self.on_update = on_update
        self.angles = None
        self.speed = None
        node.create_subscription(JointState, '/joint_states', self.update, 10)

    def update(self, msg: JointState):
        positions = dict(zip(msg.name, msg.position))
        velocities = dict(zip(msg.name, msg.velocity))
        try:
            self.angles = np.array([positions[name] for name in kin.JOINT_NAMES])
            self.speed = max(abs(velocities[name]) for name in kin.JOINT_NAMES)
        except KeyError:
            self.node.get_logger().warn('joint_states missing a joint', throttle_duration_sec=1.0)
            return
        if self.on_update:
            self.on_update(msg)


class LaunchKey:
    """The current episode's launch key. `on_new_episode` runs for every key after the first, i.e. on a reset."""

    def __init__(self, node, on_new_episode):
        self.value = None
        self.on_new_episode = on_new_episode
        node.create_subscription(String, '/sim/launch_key', self.update, LATCHED)

    def update(self, msg: String):
        new_episode = self.value is not None
        self.value = msg.data
        if new_episode:
            self.on_new_episode()


class Frames:
    """TF frames as panels in the world, refusing a transform whose publisher has gone quiet."""

    def __init__(self, node, max_age=0.5):
        self.clock = node.get_clock()
        self.max_age = max_age
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, node)

    def panel(self, frame):
        """The frame's pose in the world right now."""
        try:
            transform = self.buffer.lookup_transform(WORLD_FRAME, frame, Time())
        except tf2_ros.TransformException as err:
            raise PanelUnavailable(f'no {frame} transform: {err}') from None

        age = elapsed(self.clock, Time.from_msg(transform.header.stamp))
        if age > self.max_age:
            raise PanelUnavailable(f'{frame} transform is {age:.1f} s old, its publisher has stopped')

        t, q = transform.transform.translation, transform.transform.rotation
        return Panel(np.array([t.x, t.y, t.z]), Rotation.from_quat([q.x, q.y, q.z, q.w]))


def run(node_class, args=None):
    rclpy.init(args=args)
    node = node_class()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
