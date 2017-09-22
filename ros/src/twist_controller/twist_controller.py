import math
import rospy
import numpy as np
from pid import PID
from yaw_controller import YawController
import tf

ONE_MPH = 0.44704
POLYNOMIAL_ORDER = 3


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

        #self.throttle_pid = PID(kp=1.0, ki=0.0, kd=0.0, mn=0, mx=accel_limit)
        #self.steering_pid = PID(kp=0.1, ki=0.0003, kd=0.1, mn=-max_steer_angle, mx=max_steer_angle)
        self.throttle_pid = PID(kp=1.0, ki=0.0, kd=0.0, mn=0, mx=accel_limit)
        self.steering_pid = PID(kp=0.1, ki=0.0, kd=0.0, mn=-max_steer_angle, mx=max_steer_angle)
        self.timestamp = rospy.get_time()


    def control(self, waypoints, pose, target_linear_vel, target_angular_vel, current_linear_vel):

        current_time = rospy.get_time()
        sample_time = current_time - self.timestamp
        self.timestamp = current_time

        velocity_error = target_linear_vel - current_linear_vel
        throttle = self.throttle_pid.step(velocity_error, sample_time)

        # TODO: <=?
        if throttle < 0:
            brake = self.wheel_radius * self.vehicle_mass * abs(self.decel_limit)
            throttle = 0
        else:
            brake = 0

        cte = self.cross_track_error(waypoints, pose)
        steer = self.steering_pid.step(cte, sample_time)
        yaw_steer = self.yaw_controller.get_steering(target_linear_vel, target_angular_vel, current_linear_vel)


        #return throttle, brake, steer + yaw_steer
        return throttle, brake,  yaw_steer


    def cross_track_error(self, waypoints, pose):
        if not waypoints or not pose:
            return 0

        pose_x = pose.position.x
        pose_y = pose.position.y
        lane_x = []
        lane_y = []

        quaternion = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = -euler[2]

        for i in range(len(waypoints)):
            temp_x = waypoints[i].pose.pose.position.x - pose_x
            temp_y = waypoints[i].pose.pose.position.y - pose_y
            # TODO: yaw or -yaw?
            lane_x.append(temp_x * math.cos(yaw) - temp_y * math.sin(yaw))
            lane_y.append(temp_x * math.sin(yaw) + temp_y * math.cos(yaw))

        polynomial = np.polyfit(lane_x, lane_y, POLYNOMIAL_ORDER)
        target_y = np.polyval(polynomial, 0.0 )
        
        #return target_y #- pose_y ???
        return target_y
