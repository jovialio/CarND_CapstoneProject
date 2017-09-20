from styx_msgs.msg import TrafficLight
import cv2
import rospy
import tensorflow as tf
from keras import backend as K
from tensorflow.core.framework import graph_pb2
from time import time

class TLClassifier(object):
    def __init__(self):
        K.clear_session()
        self.sess = tf.Session()
        self.graph = self.sess.graph

        with open('classifier_simu.pb', "rb") as f:
            output_graph_def = graph_pb2.GraphDef()
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
            self.image_input = self.sess.graph.get_tensor_by_name('input_1:0')
            self.softmax = self.sess.graph.get_tensor_by_name('output_node0:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Resize image to what the model is trained on
        img_shape = (224, 224, 3)

        with self.graph.as_default():
            start = time()
            img = cv2.resize(image, (img_shape[0], img_shape[1]), interpolation=cv2.INTER_AREA)

            preds = self.sess.run(self.softmax, {self.image_input: img[None, :, :, :]})

            state = preds.argmax()
            prob = preds[state]

            rospy.logwarn('state ' + str(state))
            rospy.logwarn('All preds ' + str(preds))

            rospy.logwarn('classification time is ' + str((time() - start) * 1000))
            if state == 1:
                return TrafficLight.RED
            else:
                return TrafficLight.UNKNOWN