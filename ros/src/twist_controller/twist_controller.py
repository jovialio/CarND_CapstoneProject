import math
import rospy
import numpy as np
from pid import PID
from yaw_controller import YawController

ONE_MPH = 0.44704
POLYNOMIAL_ORDER = 3


class Controller(object):
    def __init__(self, *args, **kwargs):

        decel_limit = kwargs['decel_limit']
        accel_limit = kwargs['accel_limit']
        wheel_base = kwargs['wheel_base']
        steer_ratio = kwargs['steer_ratio']
        min_speed = kwargs['min_speed']
        max_lat_accel = kwargs['max_lat_accel']
        max_steer_angle = kwargs['max_steer_angle']

        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        self.throttle_pid = PID(kp=1.0, ki=0.0, kd=0.0, mn=decel_limit, mx=accel_limit)
        self.steering_pid = PID(kp=0.1, ki=0.0003, kd=0.1, mn=-max_steer_angle, mx=max_steer_angle)
        self.timestamp = rospy.get_time()


    def control(self, waypoints, pose, target_linear_vel, target_angular_vel, current_linear_vel):

        current_time = rospy.get_time()
        sample_time = current_time - self.timestamp
        self.timestamp = current_time

        velocity_error = target_linear_vel - current_linear_vel
        throttle = self.throttle_pid.step(velocity_error, sample_time)

        if throttle < 0:
            brake = throttle
            throttle = 0
        else:
            brake = 0

        cte = self.cross_track_error(waypoints, pose)
        steer = self.steering_pid.step(cte, sample_time)
        yaw_steer = self.yaw_controller.get_steering(target_linear_vel, target_angular_vel, current_linear_vel)

        return throttle, brake, steer + yaw_steer


    def cross_track_error(self, waypoints, pose):
        if not waypoints or not pose:
            return 0

        lane_x = [waypoint.pose.pose.position.x for waypoint in waypoints]
        lane_y = [waypoint.pose.pose.position.y for waypoint in waypoints]
        pose_x = pose.position.x
        pose_y = pose.position.y

        polynomial = np.polyfit(lane_x, lane_y, POLYNOMIAL_ORDER)
        target_y = np.polyval(polynomial, pose_x)
        
        return target_y - pose_y
