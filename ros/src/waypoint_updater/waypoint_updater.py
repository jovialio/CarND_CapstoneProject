#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32


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
CRUISE_VELOCITY = 20 # 9 for roughly 20 MPH in M/S


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # do these have similar cycles?
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        #rospy.Subscriber('/obstacle_waypoint', TBD, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        self.base_waypoints = None
        self.current_pose = None

        self.loop()

        rospy.spin()

    def pose_cb(self, msg):
        """Receives a PoseStamped message containing a Header and a Pose"""
        rospy.loginfo(rospy.get_name() + ': pose received')
        self.current_pose = msg.pose

    def waypoints_cb(self, msg):
        """Receives a Lane message containing a Header and Waypoints"""
        rospy.loginfo(rospy.get_name() + ': waypoints received')
        self.base_waypoints = msg.waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

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
        if self.current_pose is None or self.base_waypoints is None:
            return

        closest_waypoint_index = self.get_closest_waypoint_index()
        output_waypoints = []

        for i in range(closest_waypoint_index, closest_waypoint_index + LOOKAHEAD_WPS):
            waypoint_index = i % len(self.base_waypoints)
            self.set_waypoint_velocity(self.base_waypoints, waypoint_index, CRUISE_VELOCITY)
            output_waypoints.append(self.base_waypoints[waypoint_index])

        # TODO header?
        lane = Lane()
        lane.waypoints = output_waypoints
        rospy.loginfo(rospy.get_name() + ': waypoints published')
        self.final_waypoints_pub.publish(lane)

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

        rate = rospy.Rate(10) #down from 40 due performance issues
        
        while not rospy.is_shutdown():
            self.update_waypoints()
            rate.sleep()


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
