from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='perception', executable='vision', output='screen'),
        Node(package='planning', executable='kinematics', output='screen'),
        Node(package='arm-control', executable='control', output='screen'),
        Node(package='planning', executable='type_keys', output='screen', parameters=[{'autostart': True}]),
    ])
