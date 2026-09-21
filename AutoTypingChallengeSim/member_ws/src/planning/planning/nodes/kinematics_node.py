"""Node `kinematics`: world-frame head and aim targets in, joint targets out."""
import numpy as np
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

from planning.geometry import kinematics as kin
from planning.utils.ros import WORLD_FRAME, ArmState, LaunchKey, point_array, run


class KinematicsNode(Node):
    """Turns world-frame targets into joint targets: `head_target` sets the arm joints, `aim_target` the pan/tilt."""

    def __init__(self):
        super().__init__('kinematics')

        self.arm = ArmState(self, self.joint_state_callback)
        LaunchKey(self, self.go_home)
        self.create_subscription(PointStamped, '/planning/head_target', self.head_target_callback, 10)
        self.create_subscription(PointStamped, '/planning/aim_target', self.aim_target_callback, 10)
        self.target_pub = self.create_publisher(JointState, '/target_states', 10)

        self.target_angles = None
        self.aim_point = None

    def go_home(self):
        """A new episode puts the arm back home, so drop the old target instead of driving the arm away from home."""
        self.target_angles = kin.Q_HOME.copy()
        self.aim_point = None
        self.publish_target()

    def joint_state_callback(self, msg: JointState):
        if self.target_angles is None:
            self.target_angles = self.arm.angles.copy()

        # Re-aim from the measured arm joints, which are still moving toward their target.
        if self.aim_point is not None:
            self.solve_aim()
            self.publish_target()

    def head_target_callback(self, msg: PointStamped):
        if not self.accepts(msg):
            return
        try:
            self.target_angles[:3] = kin.position_ik(point_array(msg))
        except kin.IKError as err:
            self.get_logger().warn(f'head_target rejected: {err}')
            return
        self.publish_target()

    def aim_target_callback(self, msg: PointStamped):
        if not self.accepts(msg):
            return
        self.aim_point = point_array(msg)
        self.solve_aim()
        self.publish_target()

    def accepts(self, msg: PointStamped):
        if self.arm.angles is None:
            self.get_logger().warn('no joint_states yet, ignoring target', throttle_duration_sec=1.0)
            return False
        if msg.header.frame_id != WORLD_FRAME:
            self.get_logger().warn(f"target frame is '{msg.header.frame_id}', only '{WORLD_FRAME}' is supported")
            return False
        return True

    def solve_aim(self):
        aim = kin.aim_inverse(self.arm.angles[:3], self.aim_point)
        problems = kin.aim_violations(*aim)
        if problems:
            self.get_logger().warn('aim not pressable: ' + '; '.join(problems), throttle_duration_sec=1.0)

        # Never command a joint past its hard stop.
        self.target_angles[3:] = np.clip([aim.pan, aim.tilt], kin.Q_MIN[3:], kin.Q_MAX[3:])

    def publish_target(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(kin.JOINT_NAMES)
        msg.position = self.target_angles.tolist()
        self.target_pub.publish(msg)


def main(args=None):
    run(KinematicsNode, args)


if __name__ == '__main__':
    main()
