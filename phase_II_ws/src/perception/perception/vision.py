import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R

class Vision(Node):
    def __init__(self):
        super().__init__('vision')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
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
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        r = R.from_matrix(rotation_matrix)
        q = r.as_quat()
        return q[0], q[1], q[2], q[3]
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

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
                        R_cw, _ = cv2.Rodrigues(rvec)
                        t_cw = tvec

                        R_world_cam = R_cw.T
                        t_world_cam = -R_world_cam @ t_cw

                        cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)

                        qx, qy, qz, qw = self.rotation_matrix_to_quaternion(R_world_cam)

                        transform = TransformStamped()
                        transform.header.stamp = self.get_clock().now().to_msg()
                        transform.header.frame_id = "world"
                        transform.child_frame_id = "camera"
                        transform.transform.translation.x = t_world_cam[0][0]
                        transform.transform.translation.y = t_world_cam[1][0]
                        transform.transform.translation.z = t_world_cam[2][0]

                        transform.transform.rotation.x = qx
                        transform.transform.rotation.y = qy
                        transform.transform.rotation.z = qz
                        transform.transform.rotation.w = qw

                        self.tf_broadcaster.send_transform(transform)

            else:
                self.get_logger().info('No ArUco markers detected.')

            cv2.imshow("OpenCV view", cv_image)
            cv2.waitKey(1)

        except CvBridgeError as e:
            self.get_logger().error(f'Error converting ROS Image to OpenCV: {e}')

def main(args=None):
    rclpy.init(args=args)
    vision_node = Vision()
    rclpy.spin(vision_node)
    vision_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()