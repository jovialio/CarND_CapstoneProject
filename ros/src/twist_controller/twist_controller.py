import math
import rospy
import numpy as np
from pid import PID
from yaw_controller import YawController
import tf


class Controller(object):
    def __init__(self, *args, **kwargs):

        self.decel_limit = kwargs['decel_limit']
        accel_limit = kwargs['accel_limit']
        wheel_base = kwargs['wheel_base']
        steer_ratio = kwargs['steer_ratio']
        min_speed = kwargs['min_speed']
        max_lat_accel = kwargs['max_lat_accel']
        max_steer_angle = kwargs['max_steer_angle']
        self.vehicle_mass = kwargs['vehicle_mass']
        self.wheel_radius = kwargs['wheel_radius']
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
        self.throttle_pid = PID(kp=1.0, ki=0.0, kd=0.0, mn=0, mx=accel_limit)
        self.timestamp = rospy.get_time()


    def control(self, waypoints, pose, target_linear_vel, target_angular_vel, current_linear_vel):

        current_time = rospy.get_time()
        sample_time = current_time - self.timestamp
        self.timestamp = current_time

        velocity_error = target_linear_vel - current_linear_vel

        # TODO: check what happens if sample_time is very large
        throttle = self.throttle_pid.step(velocity_error, sample_time)

        # Needs to be <= so that car actually stops, otherwise goes through lights
        if throttle <= 0:
            # TODO: why decel not proportional to velocity error ?
            brake = self.wheel_radius * self.vehicle_mass * abs(self.decel_limit)
            throttle = 0
        else:
            brake = 0

        yaw_steer = self.yaw_controller.get_steering(target_linear_vel, target_angular_vel, current_linear_vel)

        return throttle, brake,  yaw_steer


    def reset(self):
        self.throttle_pid.reset()
