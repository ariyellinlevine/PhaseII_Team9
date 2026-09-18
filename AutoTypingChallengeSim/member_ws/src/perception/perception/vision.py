import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R

class Vision(Node):
    def __init__(self):
        super().__init__('vision')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._board_lock = threading.Lock()
        self._board_n = 0
        self._board_t_sum = np.zeros(3)
        self._board_q_sum = np.zeros(4)
        self.create_timer(0.02, self.publish_board)

        # /sim/launch_key is published when a new sim episode starts, so when the world resets it runs
        self.create_subscription(
            String,
            '/sim/launch_key',
            self.launch_key_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL)
        )

        self.aruco_world_points = {
            0: np.array([
                [0.006, 0.006, 0.0],
                [0.026, 0.006, 0.0],
                [0.026, 0.026, 0.0], 
                [0.006, 0.026, 0.0]
                ], dtype=np.float32),
            1: np.array([
                [0.374, 0.006, 0.0], 
                [0.394, 0.006, 0.0], 
                [0.394, 0.026, 0.0], 
                [0.374, 0.026, 0.0]  
                ], dtype=np.float32),
            2: np.array([
                [0.374, 0.149, 0.0], 
                [0.394, 0.149, 0.0], 
                [0.394, 0.169, 0.0], 
                [0.374, 0.169, 0.0]  
                ], dtype=np.float32),
            3: np.array([
                [0.006, 0.149, 0.0], 
                [0.026, 0.149, 0.0], 
                [0.026, 0.169, 0.0], 
                [0.006, 0.169, 0.0]  
                ], dtype=np.float32)
        }

        # Define Camera Intrinsics
        fx, fy = 900.0, 900.0  # Focal lengths
        cx, cy = 639.5, 359.5  # Principal point (image center)
        W, H = 1280, 720       # Frame dimensions

        self.camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

        # no distortion
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, 'ArucoDetector'):
            # OpenCV >= 4.7
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        else:
            # OpenCV < 4.7 (e.g. the python3-opencv apt package on Ubuntu Noble)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.aruco_detector = None

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        r = R.from_matrix(rotation_matrix)
        q = r.as_quat()
        return q[0], q[1], q[2], q[3]

    def update_board_estimate(self, R_wb, t_wb):
        q = np.array(self.rotation_matrix_to_quaternion(R_wb))
        with self._board_lock:
            if np.dot(self._board_q_sum, q) < 0.0:  # q and -q are the same rotation
                q = -q
            self._board_q_sum += q
            self._board_t_sum += t_wb
            self._board_n += 1

    def launch_key_callback(self, msg):
        with self._board_lock:
            self._board_n = 0
            self._board_t_sum[:] = 0.0
            self._board_q_sum[:] = 0.0
        self.get_logger().info('New episode: remove old cachedworld -> board pose transform.')

    def publish_board(self):
        with self._board_lock:
            if self._board_n == 0:
                return
            t = self._board_t_sum / self._board_n
            q = self._board_q_sum / np.linalg.norm(self._board_q_sum)

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "world"
        transform.child_frame_id = "board"
        transform.transform.translation.x = float(t[0])
        transform.transform.translation.y = float(t[1])
        transform.transform.translation.z = float(t[2])
        transform.transform.rotation.x = float(q[0])
        transform.transform.rotation.y = float(q[1])
        transform.transform.rotation.z = float(q[2])
        transform.transform.rotation.w = float(q[3])
        self.tf_broadcaster.sendTransform(transform)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            if self.aruco_detector is not None:
                corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    gray, self.aruco_dict, parameters=self.aruco_params
                )

            if ids is not None:
                self.get_logger().info(f'Detected ArUco markers with IDs: {ids.flatten()}')

                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

                obj_points = []
                img_points = []
                
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id in self.aruco_world_points:
                        obj_points.append(self.aruco_world_points[marker_id])
                        img_points.append(corners[i].reshape(4, 2))

                if len(obj_points) > 0:
                    obj_points = np.vstack(obj_points).astype(np.float32)
                    img_points = np.vstack(img_points).astype(np.float32)

                    success, rvec, tvec = cv2.solvePnP(
                        obj_points, img_points, self.camera_matrix, self.dist_coeffs
                    )

                    if success:
                        R_cb, _ = cv2.Rodrigues(rvec)
                        t_cb = tvec.reshape(3)

                        cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)

                        try:
                            tf_wc = self.tf_buffer.lookup_transform(
                                'world', 'camera_optical_frame', Time.from_msg(msg.header.stamp),
                                timeout=Duration(seconds=0.1)
                            )
                        except TransformException as e:
                            self.get_logger().warn(
                                f'No world -> camera_optical_frame transform yet: {e}',
                                throttle_duration_sec=1.0
                            )
                        else:
                            q = tf_wc.transform.rotation
                            R_wc = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                            p = tf_wc.transform.translation
                            t_wc = np.array([p.x, p.y, p.z])

                            R_wb = R_wc @ R_cb
                            t_wb = R_wc @ t_cb + t_wc

                            # all four corners of all detected markers must be seen to update the board pose estimate
                            if len(obj_points) == 4 * len(self.aruco_world_points):
                                self.update_board_estimate(R_wb, t_wb)

            else:
                self.get_logger().info('No ArUco markers detected.')

            cv2.imshow("OpenCV view", cv_image)
            cv2.waitKey(1)

        except CvBridgeError as e:
            self.get_logger().error(f'Error converting ROS Image to OpenCV: {e}')

def main(args=None):
    rclpy.init(args=args)
    vision_node = Vision()
    executor = MultiThreadedExecutor()
    executor.add_node(vision_node)
    executor.spin()
    vision_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()