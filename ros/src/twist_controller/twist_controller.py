import math

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, decel_limit, accel_limit, max_steer_angle):

        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.max_steer_angle = max_steer_angle


    def control(self, target_linear_velocity, target_angular_velocity, current_linear_velocity, current_angular_velocity, throttle, brake):

        target_velocity = math.sqrt( target_linear_velocity*target_linear_velocity + target_angular_velocity*target_angular_velocity )
        current_velocity = math.sqrt(current_linear_velocity*current_linear_velocity + current_angular_velocity*current_angular_velocity)

        velocity_error = target_velocity - current_velocity

        if velocity_error > 0.0:
            brake = 0
            throttle = throttle + (self.accel_limit/10.0)
        elif velocity_error < 0.0:
            throttle = 0
            brake = brake + (self.decel_limit/10.0)



        # TODO: check limits

        return throttle, brake
