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

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number

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

    def clone_waypoint(self, wp):
        p = Waypoint()
        p.pose.header = wp.pose.header
        p.pose.pose.position.x = wp.pose.pose.position.x
        p.pose.pose.position.y = wp.pose.pose.position.y
        p.pose.pose.position.z = wp.pose.pose.position.z
        p.pose.pose.orientation.x = wp.pose.pose.orientation.x
        p.pose.pose.orientation.y = wp.pose.pose.orientation.y
        p.pose.pose.orientation.z = wp.pose.pose.orientation.z
        p.pose.pose.orientation.w = wp.pose.pose.orientation.w
        p.twist.header = wp.twist.header
        p.twist.twist.linear.x = wp.twist.twist.linear.x
        p.twist.twist.linear.y = wp.twist.twist.linear.y
        p.twist.twist.linear.z = wp.twist.twist.linear.z
        p.twist.twist.angular.x = wp.twist.twist.angular.x
        p.twist.twist.angular.y = wp.twist.twist.angular.y
        p.twist.twist.angular.z = wp.twist.twist.angular.z

        return p

    def update_waypoints(self):
        if self.current_pose is None or self.base_waypoints is None:
            return

        closest_waypoint_index = self.get_closest_waypoint_index()
        target_vel = self.get_waypoint_velocity(self.base_waypoints[closest_waypoint_index])
        distance_to_light = None
        output_waypoints = []
        curr_traffic_light = self.trafficlight

        if curr_traffic_light is not None and closest_waypoint_index is not None:
            distance_to_light = self.distance(self.base_waypoints, closest_waypoint_index, curr_traffic_light)
            lookehead_dist = self.distance(self.base_waypoints, closest_waypoint_index,
                                           (closest_waypoint_index + LOOKAHEAD_WPS) % len(self.base_waypoints))

        if distance_to_light is not None and \
            (self.get_num_waypoints(closest_waypoint_index, curr_traffic_light) < LOOKAHEAD_WPS):

            velocity = target_vel * distance_to_light / lookehead_dist
            if velocity < 1:
                velocity = 0;

            if curr_traffic_light - closest_waypoint_index > 2:
                output_waypoints.append(self.clone_waypoint(self.base_waypoints[closest_waypoint_index]))
                self.set_waypoint_velocity(output_waypoints, -1, velocity)

                for i in range(closest_waypoint_index + 1, curr_traffic_light - 2):
                    waypoint_index = i % len(self.base_waypoints)
                    segment_distance = self.distance(self.base_waypoints, waypoint_index - 1, waypoint_index)
                    decel_velocity = (segment_distance / distance_to_light) * self.current_velocity
                    velocity -= decel_velocity

                    if velocity < 1:
                        velocity = 0

                    output_waypoints.append(self.clone_waypoint(self.base_waypoints[waypoint_index]))
                    self.set_waypoint_velocity(output_waypoints, -1, velocity)

            if curr_traffic_light == closest_waypoint_index:
                output_waypoints.append(self.clone_waypoint(self.base_waypoints[curr_traffic_light]))
                self.set_waypoint_velocity(output_waypoints, -1, 0)

            else:
                output_waypoints.append(self.clone_waypoint(self.base_waypoints[curr_traffic_light - 1]))
                self.set_waypoint_velocity(output_waypoints, -1, 0)
                output_waypoints.append(self.clone_waypoint(self.base_waypoints[curr_traffic_light]))
                self.set_waypoint_velocity(output_waypoints, -1, 0)

            for i in range(curr_traffic_light, closest_waypoint_index + LOOKAHEAD_WPS):
                waypoint_index = i % len(self.base_waypoints)
                output_waypoints.append(self.clone_waypoint(self.base_waypoints[waypoint_index]))

        else:
            for i in range(closest_waypoint_index, closest_waypoint_index + LOOKAHEAD_WPS):
                waypoint_index = i % len(self.base_waypoints)
                output_waypoints.append(self.clone_waypoint(self.base_waypoints[waypoint_index]))


        lane = Lane()
        self.final_waypoints = output_waypoints
        lane.waypoints = output_waypoints
        self.final_waypoints_pub.publish(lane)

    def get_num_waypoints(self, start, finish):
        if start < finish:
            return finish - start
        else:
            return len(self.base_waypoints) - start + finish

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


    def loop(self):

        rate = rospy.Rate(40) #down from 40 due performance issues
        
        while not rospy.is_shutdown():
            self.update_waypoints()
            rate.sleep()


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
