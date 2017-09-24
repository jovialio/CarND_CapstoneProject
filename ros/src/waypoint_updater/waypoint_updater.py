#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
CRUISE_VELOCITY = 4.4704 # 9 for roughly 20 MPH in M/S
TRAFFIC_LIGHT_RANGE = 15

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # do these have similar cycles?
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_vel_cb)

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        self.base_waypoints = None
        self.current_pose = None
        self.max_x_pos = None
        self.trafficlight = None
        self.final_waypoints = None
        self.current_velocity = None

        self.loop()

        rospy.spin()

    def current_vel_cb(self, msg):
        self.current_velocity = msg.twist.linear.x

    def pose_cb(self, msg):
        """Receives a PoseStamped message containing a Header and a Pose"""
        rospy.loginfo(rospy.get_name() + ': pose received')
        self.current_pose = msg.pose

    def waypoints_cb(self, msg):
        """Receives a Lane message containing a Header and Waypoints"""
        rospy.loginfo(rospy.get_name() + ': waypoints received')
        self.base_waypoints = msg.waypoints
        if not self.max_x_pos:
            waypoints_x = [waypoint.pose.pose.position.x for waypoint in self.base_waypoints]
            self.max_x_pos = np.max(np.array(waypoints_x))

    def traffic_cb(self, msg):
        if msg.data == -1:
            self.trafficlight = None
        else:
            self.trafficlight = msg.data

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def update_waypoints(self):
        if self.current_pose is None or self.base_waypoints is None\
                or self.max_x_pos is None:
            return

        closest_waypoint_index = self.get_closest_waypoint_index()
        distance_to_light = None
        output_waypoints = []

        if self.trafficlight is not None and closest_waypoint_index is not None:
            distance_to_light = self.distance(self.base_waypoints, closest_waypoint_index, self.trafficlight)

        if distance_to_light is not None and distance_to_light < TRAFFIC_LIGHT_RANGE:

            velocity = CRUISE_VELOCITY * distance_to_light / TRAFFIC_LIGHT_RANGE
            if velocity < 1:
                velocity = 0;

            if self.trafficlight - closest_waypoint_index > 2:
                self.set_waypoint_velocity(self.base_waypoints, closest_waypoint_index, velocity)
                output_waypoints.append(self.base_waypoints[closest_waypoint_index])

                for i in range(closest_waypoint_index + 1, self.trafficlight - 2):
                    waypoint_index = i % len(self.base_waypoints)
                    segment_distance = self.distance(self.base_waypoints, waypoint_index - 1, waypoint_index)
                    decel_velocity = (segment_distance / distance_to_light) * self.current_velocity
                    velocity -= decel_velocity

                    if velocity < 1:
                        velocity = 0

                    self.set_waypoint_velocity(self.base_waypoints, waypoint_index, velocity)
                    output_waypoints.append(self.base_waypoints[waypoint_index])

            if self.trafficlight == closest_waypoint_index:
                self.set_waypoint_velocity(self.base_waypoints, self.trafficlight, 0)
                output_waypoints.append(self.base_waypoints[self.trafficlight])

            else:
                self.set_waypoint_velocity(self.base_waypoints, self.trafficlight - 1, 0)
                output_waypoints.append(self.base_waypoints[self.trafficlight - 1])
                self.set_waypoint_velocity(self.base_waypoints, self.trafficlight, 0)
                output_waypoints.append(self.base_waypoints[self.trafficlight])

        else:
            for i in range(closest_waypoint_index, closest_waypoint_index + LOOKAHEAD_WPS):
                waypoint_index = i % len(self.base_waypoints)
                self.set_waypoint_velocity(self.base_waypoints, waypoint_index, CRUISE_VELOCITY)
                output_waypoints.append(self.base_waypoints[waypoint_index])

        lane = Lane()
        self.final_waypoints = output_waypoints
        lane.waypoints = output_waypoints
        rospy.loginfo(rospy.get_name() + ': waypoints published')
        self.final_waypoints_pub.publish(lane)

    def normalize(self, x):
        return self.max_x_pos + x % self.max_x_pos

    def get_closest_waypoint_index(self):
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        min_idx = None
        min_dist = None

        for i in range(len(self.base_waypoints)):
            dist = dl(self.base_waypoints[i].pose.pose.position, self.current_pose.position)
            if not min_dist or min_dist > dist:
                min_idx = i
                min_dist = dist

        return min_idx

    def get_closest_waypoint_ahead_index(self):
        dl = lambda a, b: math.sqrt((self.normalize(a.x) - self.normalize(b.x)) ** 2 +
                                    (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        min_idx = None
        min_dist = None

        for i in range(len(self.base_waypoints)):
            dist = dl(self.base_waypoints[i].pose.pose.position, self.current_pose.position)
            if not min_dist or min_dist > dist and \
                            self.normalize(self.base_waypoints[i].pose.pose.position.x) > \
                            self.normalize(self.current_pose.position.x):
                min_idx = i
                min_dist = dist
        return min_idx

    def loop(self):

        rate = rospy.Rate(10) #down from 40 due performance issues
        
        while not rospy.is_shutdown():
            self.update_waypoints()
            rate.sleep()


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
