# Copyright (C) 2025 Dan Novischi. All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v3 or any later version.

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from antrobot_ros.utils import load_node_params


def generate_launch_description():
    robot_namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace for the robot instance'
    )

    config_file_path = os.path.join(
        get_package_share_directory('antrobot_ros'),
        'config',
        'antrobot_params.yaml'
    )

    camera_params = load_node_params(config_file_path, 'camera')
    web_video_params = load_node_params(config_file_path, 'web_video_server')

    v4l2_camera_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        namespace=LaunchConfiguration('namespace'),
        name='v4l2_camera',
        output='screen',
        parameters=[{
            'video_device': camera_params.get('video_device', '/dev/video0'),
            'image_size': camera_params.get('image_size', [640, 480]),
            'pixel_format': camera_params.get('pixel_format', 'YUYV'),
            'camera_frame_id': camera_params.get('camera_frame_id', 'camera_link'),
        }],
        remappings=[
            ('image_raw', camera_params.get('image_topic', 'camera/image_raw')),
        ],
    )

    web_video_server_node = Node(
        package='web_video_server',
        executable='web_video_server',
        namespace=LaunchConfiguration('namespace'),
        name='web_video_server',
        output='screen',
        parameters=[{
            'port': web_video_params.get('port', 8080),
            'server_threads': web_video_params.get('server_threads', 1),
            'ros_threads': web_video_params.get('ros_threads', 2),
            'default_stream_type': web_video_params.get('default_stream_type', 'mjpeg'),
        }],
    )

    return LaunchDescription([
        robot_namespace_arg,
        v4l2_camera_node,
        web_video_server_node,
    ])
