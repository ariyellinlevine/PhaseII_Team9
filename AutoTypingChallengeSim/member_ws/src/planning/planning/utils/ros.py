
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

WORLD_FRAME = 'world'
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def point_msg(point, clock):
    msg = PointStamped()
    msg.header.stamp = clock.now().to_msg()
    msg.header.frame_id = WORLD_FRAME
    msg.point.x, msg.point.y, msg.point.z = point
    return msg


def point_array(msg: PointStamped):
    return np.array([msg.point.x, msg.point.y, msg.point.z])


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
