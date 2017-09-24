#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 3
LOOKAHEAD_WAYPOINTS = 50

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.base_waypoints = None
        self.camera_image = None
        self.lights = []
        self.close_waypoints = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        self.traffic_lights_sub = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()

        self.classifier = rospy.get_param("/classifier")
        self.simulator = rospy.get_param("/simulator")

        self.light_classifier = TLClassifier(self.classifier, self.simulator)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.light_waypoints = []
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, msg):
        self.base_waypoints = msg.waypoints
        stop_positions = self.config['stop_line_positions']

        for stop_pos in stop_positions:
            pos = PoseStamped()
            pos.pose.position.x = stop_pos[0]
            pos.pose.position.y = stop_pos[1]

            stop_waypoint_idx = self.get_closest_waypoint(pos.pose)
            self.close_waypoints.append((stop_waypoint_idx, stop_pos))

        sorted(self.close_waypoints)
        self.base_waypoints_sub.unregister()

    def traffic_cb(self, msg):
        if self.base_waypoints:
            self.lights = msg.lights
            for light in self.lights:
                light_waypoint_idx = self.get_closest_waypoint(light.pose.pose)
                self.light_waypoints.append((light_waypoint_idx, light.pose))

            sorted(self.light_waypoints)
            self.traffic_lights_sub.unregister()

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        min_idx = None
        min_dist = None

        for i in range(len(self.base_waypoints)):
            dist = dl(self.base_waypoints[i].pose.pose.position, pose.position)
            if not min_dist or min_dist > dist:
                min_idx = i
                min_dist = dist

        return min_idx

    def get_closest_waypoint_idx(self, waypoints, reference_idx):
        min_idx = None
        for wp in waypoints:
            end_idx = (reference_idx + LOOKAHEAD_WAYPOINTS) % len(self.base_waypoints)
            if reference_idx > end_idx:
                if wp[0] > reference_idx or wp[0] < end_idx:
                    min_idx = wp[0]
                    break
            else:
                if reference_idx < wp[0] < end_idx:
                    min_idx = wp[0]
                    break
        return min_idx

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """
        x = 0
        y = 0

        #TODO Use tranform and rotation to calculate 2D position of light in image
        return (x, y)

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        self.camera_image.encoding = "rgb8"
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        x, y = self.project_to_image_plane(light.pose.pose.position)

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if self.base_waypoints and self.pose and self.light_waypoints:
            car_waypoint_idx = self.get_closest_waypoint(self.pose.pose)
            light = None
            min_light_idx = self.get_closest_waypoint_idx(self.light_waypoints, car_waypoint_idx)
            min_close_idx = None

            if min_light_idx:
                light = TrafficLight()
                light.pose = self.base_waypoints[min_light_idx].pose
                min_close_idx = self.get_closest_waypoint_idx(self.close_waypoints, car_waypoint_idx)

            if light:
                state = self.get_light_state(light)
                return min_close_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
