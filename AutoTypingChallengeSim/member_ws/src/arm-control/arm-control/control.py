import numpy as np
import rclpy
from rclpy.node import Node
import control as ct
from sensor_msgs import JointState
from autotype_msgs import JointVelocityCommand

class control(Node):
    def __init__(self):
        super().__init__('control')

        self.joint_State_subsci = self.create_subscription(
            JointState,
            'joint_states',
            self.listener_callback,
            10)

        self.publisher_ = self.create_publisher(JointVelocityCommand, '/arm/cmd_joint_velocity', 10)
        
        # Define the state-space representation of the system (A, B, C, D matrices)
        A = np.block([
        [np.zeros((5, 5)), np.eye(5)],
        [np.zeros((5, 5)), np.zeros((5, 5))]
        ])

        B = np.block([
        [np.zeros((5, 5))],
        [np.eye(5)]
        ])

        C = np.eye(10)
        D = np.zeros((10, 5))

        self.sys = ct.ss(A, B, C, D)

        # Define the state error weighting matrix (How much to penalize the state error, must be symmetric and positive definite)
        self.Q = np.diag([10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 2.78, 2.78, 1.56, 1.00, 1.00])
        self.R = np.diag([0.444, 0.444, 0.250, 0.111, 0.111])

        # Max joint acceleration ~ order: base_yaw, shoulder_pitch, elbow_pitch, head_pan, head_tilt
        self.a_max = np.array([1.5, 1.5, 2.0, 3.0, 3.0])

        """ 
        Compute the optimal gain matrix K using LQR, placeholder _ for 
        the solution of the Riccati equation (crazy cool to walk through the derivation btw) 
        and the eigenvalues of the closed-loop system cause we don't need them.
        """

        self.K, _, _ = ct.lqr(self.sys, self.Q, self.R)
        
    def control_arm(self,  current_angles, current_velocities, target_angles):

        pos_error = np.array(current_angles) - np.array(target_angles)
        #we want velocity to eventually be zero
        vel_error = np.array(current_velocities) - np.zeros(5)

        x_error = np.concatenate([pos_error, vel_error])

        # Compute the control input vector (K is the gain matrix, joint_angles is the state vector)
        control_out = -self.K @ x_error

        # Enforce the acceleration limits
        control_out = np.clip(control_out, -self.a_max, self.a_max)

        return control_out

    