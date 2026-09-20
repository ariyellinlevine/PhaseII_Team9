import numpy as np
import rclpy
from rclpy.node import Node
import control as ct
from sensor_msgs.msg import JointState
from autotype_msgs.msg import JointVelocityCommand

JOINT_NAMES = ['base_yaw', 'shoulder_pitch', 'elbow_pitch', 'head_pan', 'head_tilt']
CONTROL_PERIOD = 0.02 

class control(Node):
    def __init__(self):
        super().__init__('control')
        
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.create_subscription(
            JointState,
            '/target_states',
            self.target_state_callback,
            10
        )

        self.joint_command_pub = self.create_publisher(
            JointVelocityCommand,
            '/arm/cmd_joint_velocity',
            1
        )

        self.have_joint_state = False
        self.create_timer(CONTROL_PERIOD, self.control_arm)

        # Define the state-space representation of the system (A, B, C, D matrices)
        A = np.block([
        [np.zeros((5, 5)), np.eye(5)],
        [np.zeros((5, 5)), np.zeros((5, 5))]
        ])

        B = np.block([
        [np.zeros((5, 5))],
        [np.eye(5)]
        ])

        self.current_angles = np.zeros(5)
        self.current_velocities = np.zeros(5)
        self.target_angles = np.zeros(5)

        C = np.eye(10)
        D = np.zeros((10, 5))

        self.sys = ct.ss(A, B, C, D)

        # Define the state error weighting matrix (How much to penalize the state error, must be symmetric and positive definite)
        self.Q = np.diag([10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 2.78, 2.78, 1.56, 1.00, 1.00])
        self.R = np.diag([0.444, 0.444, 0.250, 0.111, 0.111])

        # Max joint acceleration || order: base_yaw, shoulder_pitch, elbow_pitch, head_pan, head_tilt
        self.a_max = np.array([1.5, 1.5, 2.0, 3.0, 3.0])

        """ 
        Compute the optimal gain matrix K using LQR, placeholder _ for 
        the solution of the Riccati equation (crazy cool to walk through the derivation btw) 
        and the eigenvalues of the closed-loop system cause we don't need them.
        """

        self.K, _, _ = ct.lqr(self.sys, self.Q, self.R)
    
    def target_state_callback(self, msg: JointState):
        id = {name: i for i, name in enumerate(msg.name)}
        try:
            self.target_angles = np.array([msg.position[id[n]] for n in JOINT_NAMES])
        except (KeyError, IndexError):
            self.get_logger().warn('target_states missing a joint or field', throttle_duration_sec=1.0)
            return


    def joint_state_callback(self, msg: JointState):
        id = {name: i for i, name in enumerate(msg.name)}
        try:
            self.current_angles = np.array([msg.position[id[n]] for n in JOINT_NAMES])
            self.current_velocities = np.array([msg.velocity[id[n]] for n in JOINT_NAMES])
        except (KeyError, IndexError):
            self.get_logger().warn('joint_states missing a joint or field', throttle_duration_sec=1.0)
            return

        if not self.have_joint_state:
            self.target_angles = self.current_angles.copy()
            self.have_joint_state = True

    def control_arm(self):
        if not self.have_joint_state:
            return

        pos_error = np.array(self.current_angles) - np.array(self.target_angles)
        #we want velocity to eventually be zero
        vel_error = np.array(self.current_velocities) - np.zeros(5)

        x_error = np.concatenate([pos_error, vel_error])
        
        control_out = -self.K @ x_error

        control_out = np.clip(control_out, -self.a_max, self.a_max)

        velocity_cmd = self.current_velocities + control_out * CONTROL_PERIOD

        joint_commands = JointVelocityCommand()
        joint_commands.header.stamp = self.get_clock().now().to_msg()
        joint_commands.name = list(JOINT_NAMES)
        joint_commands.velocity = velocity_cmd.tolist()
        self.joint_command_pub.publish(joint_commands)

def main(args=None):
    rclpy.init(args=args)
    node = control()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
