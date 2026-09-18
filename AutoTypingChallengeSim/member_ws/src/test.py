import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from autotype_msgs.msg import JointVelocityCommand  # replace with actual package/msg name

class ArmVelocityPublisher(Node):
    def __init__(self):
        super().__init__('arm_velocity_publisher')
        self.publisher_ = self.create_publisher(
            JointVelocityCommand, '/arm/cmd_joint_velocity', 10)
        timer_period = 0.02  # 50 Hz — check INTERFACES.md for watchdog window
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = JointVelocityCommand()

        msg.name = ['base_yaw', 'shoulder_pitch', 'elbow_pitch', 'head_pan', 'head_tilt']
        msg.velocity = [1, 0.0, 0.0, 0.0, 0.0]  # rad/s, one per name, same order           # rad/s
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ArmVelocityPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()