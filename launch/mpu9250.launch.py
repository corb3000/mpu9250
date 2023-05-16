import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    #get_package_share_directory(package_name)
    config = os.path.join(
        get_package_share_directory('mpu9250'),
        'config',
        'imu_mpu9250.yaml'
        )

    return LaunchDescription([
        Node(
            package="mpu9250",
            executable="mpu9250",
            # name="mpu9250",
            parameters=[config]
        )
    ])
