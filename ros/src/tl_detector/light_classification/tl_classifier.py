from styx_msgs.msg import TrafficLight
import cv2
import rospy
import tensorflow as tf
from keras import backend as K
from tensorflow.core.framework import graph_pb2
from time import time
import numpy as np
import os

class TLClassifier(object):
    def __init__(self, classifier):
        self.classifier = classifier
        self.index = 0
        K.clear_session()
        self.sess = tf.Session()
        self.graph = self.sess.graph

        # if not os.path.isdir('output'):
        #     os.mkdir('output')
        # if not os.path.isdir('output' + '/0'):
        #     os.mkdir('output' + '/0')
        # if not os.path.isdir('output' + '/1'):
        #     os.mkdir('output' + '/1')

        with open(self.classifier, "rb") as f:
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
        with self.graph.as_default():
            start = time()
            img = self.preprocess(image)

            preds = self.sess.run(self.softmax, {self.image_input: img[None, :, :, :]})

            state = np.argmax(preds)
            #cv2.imwrite(os.path.join('output' + '/' + str(state), str(self.index) + ".jpg"), img)
            self.index = self.index + 1
            if state == 1:
                rospy.logwarn("Light detected - RED")
                return TrafficLight.RED
            else:
                rospy.logwarn("No Light detected")
                return TrafficLight.UNKNOWN

    def preprocess(self, image):
        # Resize image to what the model is trained on
        img_shape = (224, 224, 3)
        img = cv2.resize(image, (img_shape[0], img_shape[1]), interpolation=cv2.INTER_AREA)
        return img
