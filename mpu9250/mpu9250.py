import os
import sys
import time
import smbus
import numpy as np

from .imusensor.MPU9250 import MPU9250
from .imusensor.filters import kalman 
from .imusensor.filters import madgwick

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu, MagneticField
from rclpy.parameter import Parameter

from math import sin, cos, radians
import tf_transformations


class MyPythonNode(Node):
    def __init__(self):
        super().__init__("imu_mpu9250")
        self.declare_parameters(
            namespace='',
            parameters=[
                ('imu_pub_mag', True),
                ('imu_pub_raw', True),
                ('imu_pub_data', True),
                ('cal_mag', False),
                ('cal_accel', False),
                ('imu_sample_frequency', 100),
                ('imu_frame_id', 'imu_link'),
                ('imu_i2c_address', 0x68),
                ('imu_i2c_port', 1),
                ('acceleration_scale', [1.0, 1.0, 1.0]),
                ('acceleration_bias', [0.0, 0.0, 0.0]),
                ('gyro_bias', [0.0, 0.0, 0.0]),
                ('magnetometer_scale', [1.0, 1.0, 1.0]),
                ('magnetometer_bias', [0.0, 0.0, 0.0]),
                ('magnetometer_transform', [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]),
                ('imu_output_transform', [

                    
                    0.0, 1.0, 0.0,
                    1.0, 0.0, 0.0,
                    0.0, 0.0, -1.0]),
            ]
        )

        address = self.get_parameter('imu_i2c_address')._value
        bus = smbus.SMBus(self.get_parameter('imu_i2c_port')._value)
        self.imu = MPU9250.MPU9250(bus, address)

        self.imu.Accels = np.asarray(self.get_parameter('acceleration_scale')._value)
        self.imu.AccelBias = np.asarray(self.get_parameter('acceleration_bias')._value)
        self.imu.GyroBias = np.asarray(self.get_parameter('gyro_bias')._value)
        self.imu.Mags = np.asarray(self.get_parameter('magnetometer_scale')._value)
        self.imu.MagBias = np.asarray(self.get_parameter('magnetometer_bias')._value)
        self.imu.Magtransform = np.reshape(np.asarray(self.get_parameter('magnetometer_transform')._value),(3,3))
        self.imu.OutputTransform = np.reshape(np.asarray(self.get_parameter('imu_output_transform')._value),(3,3))
        self.pub_mag_ = self.get_parameter('imu_pub_raw')._value
        self.pub_data_ = self.get_parameter('imu_pub_data')._value
        self.pub_raw_ = self.get_parameter('imu_pub_raw')._value
        self.sample_frequency_ = self.get_parameter('imu_sample_frequency')._value

        if self.pub_raw_:
            self.publisher_imu_raw_ = self.create_publisher(Imu, "/imu/data_raw", 10)
        if self.pub_data_:
            self.publisher_imu_data_ = self.create_publisher(Imu, "/imu/data/imu", 10)
        if self.pub_mag_:
            self.publisher_imu_mag_ = self.create_publisher(MagneticField, "/imu/mag", 10)

        print("IMU Node started")

        if self.get_parameter('cal_accel')._value:
            self.imu.begin()
            self.imu.caliberateAccelerometer() # calibrate ACC

        if self.get_parameter('cal_mag')._value:
            self.imu.begin()
            self.imu.caliberateMagPrecise() # calibrate mag

        if self.get_parameter('cal_accel')._value or self.get_parameter('cal_mag')._value:
            self.imu.saveCalibDataToFile('/home/bigshark/dev_ws/imu_cal.json')
            
            update_param = [
                Parameter('acceleration_scale', Parameter.Type.DOUBLE_ARRAY, self.imu.Accels.tolist()),
                Parameter('acceleration_bias', Parameter.Type.DOUBLE_ARRAY, self.imu.AccelBias.tolist()),
                Parameter('gyro_bias', Parameter.Type.DOUBLE_ARRAY, self.imu.GyroBias.tolist()),
                Parameter('magnetometer_scale', Parameter.Type.DOUBLE_ARRAY, self.imu.Mags.tolist()),
                Parameter('magnetometer_bias', Parameter.Type.DOUBLE_ARRAY, self.imu.MagBias.tolist()),
                Parameter('magnetometer_transform', Parameter.Type.DOUBLE_ARRAY, self.imu.Magtransform.reshape(9).tolist())
            ]

            self.set_parameters(update_param)



        self.timer_publish_imu_values_ = self.create_timer(1.0/self.sample_frequency_, self.publish_imu_values)


        self.sensorfusion = kalman.Kalman()
        # self.sensorfusion = madgwick.Madgwick(0.5)
        self.imu.begin()
        self.imu.readSensor()    
        self.imu.computeOrientation()

        self.sensorfusion.roll = self.imu.roll
        self.sensorfusion.pitch = self.imu.pitch
        self.sensorfusion.yaw = self.imu.yaw
        self.deltaTime = 0
        self.lastTime = self.get_clock().now()

    def publish_imu_values(self):
        msg_data = Imu()
        msg_mag = MagneticField()

        self.imu.readSensor()
        time_stamp = self.get_clock().now().to_msg()       
        deltaTime = (self.get_clock().now() - self.lastTime).nanoseconds * 10e9
        self.lastTime = self.get_clock().now()

        msg_data.header.stamp = time_stamp
        msg_data.header.frame_id = self.get_parameter('imu_frame_id')._value
        # Direct measurements
        self.imu.AccelVals = self.imu.OutputTransform.dot(self.imu.AccelVals)
        self.imu.GyroVals = self.imu.OutputTransform.dot(self.imu.GyroVals)
        self.imu.MagVals = self.imu.OutputTransform.dot(self.imu.MagVals)
        msg_data.linear_acceleration_covariance = [0.0025, 0.0, 0.0, 0.0, 0.0025, 0.0, 0.0, 0.0, 0.0025]
        msg_data.linear_acceleration.x = self.imu.AccelVals[0]
        msg_data.linear_acceleration.y = self.imu.AccelVals[1]
        msg_data.linear_acceleration.z = self.imu.AccelVals[2]
        msg_data.angular_velocity_covariance = [0.0025, 0.0, 0.0, 0.0, 0.0025, 0.0, 0.0, 0.0, 0.0025]
        msg_data.angular_velocity.x = (self.imu.GyroVals[0]) 
        msg_data.angular_velocity.y = (self.imu.GyroVals[1]) 
        msg_data.angular_velocity.z = (self.imu.GyroVals[2])

        if self.pub_raw_: 
            self.publisher_imu_raw_.publish(msg_data)

        if self.pub_data_:        
            self.sensorfusion.computeAndUpdateRollPitchYaw(\
                self.imu.AccelVals[0], self.imu.AccelVals[1], self.imu.AccelVals[2],\
                self.imu.GyroVals[0], self.imu.GyroVals[1], self.imu.GyroVals[2],\
                self.imu.MagVals[0], self.imu.MagVals[1], self.imu.MagVals[2], deltaTime)
            yaw = self.sensorfusion.yaw
            pitch = self.sensorfusion.pitch
            roll = self.sensorfusion.roll

            # Calculate euler angles, convert to quaternion and store in message
            msg_data.orientation_covariance = [0.0025, 0.0, 0.0, 0.0, 0.0025, 0.0, 0.0, 0.0, 0.0025]
            # Convert to quaternion
            quat = tf_transformations.quaternion_from_euler(
                radians(roll), radians(pitch), radians(yaw))
            msg_data.orientation.x = quat[0]
            msg_data.orientation.y = quat[1]
            msg_data.orientation.z = quat[2]
            msg_data.orientation.w = quat[3]
            self.publisher_imu_data_.publish(msg_data)
        # print("roll: {:4.2f} \tpitch : {:4.2f} \tyaw : {:4.2f}".format(self.sensorfusion.roll, self.sensorfusion.pitch, self.sensorfusion.yaw))

        if self.pub_mag_: 
            msg_mag.header.stamp = time_stamp
            msg_mag.header.frame_id = self.get_parameter('imu_frame_id')._value            
            msg_mag.magnetic_field.x = (self.imu.MagVals[0])
            msg_mag.magnetic_field.y = (self.imu.MagVals[1])
            msg_mag.magnetic_field.z = (self.imu.MagVals[2])
            msg_mag.magnetic_field_covariance = [0.0025, 0.0, 0.0, 0.0, 0.0025, 0.0, 0.0, 0.0, 0.0025]
            self.publisher_imu_mag_.publish(msg_mag)


def main(args=None):
    rclpy.init(args=args)
    node = MyPythonNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()