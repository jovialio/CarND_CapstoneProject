from styx_msgs.msg import TrafficLight
import cv2
import rospy
import tensorflow as tf
from keras import backend as K
from tensorflow.core.framework import graph_pb2
from time import time
import numpy as np


class TLClassifier(object):
    def __init__(self, classifier, simulator):
        self.classifier = classifier
        self.simulator = simulator

        K.clear_session()
        self.sess = tf.Session()
        self.graph = self.sess.graph

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

            state = preds.argmax()
            prob = preds[state]

            if state == 1:
                return TrafficLight.RED
            else:
                return TrafficLight.UNKNOWN

    def preprocess(self, image):
        if self.simulator:
            # Resize image to what the model is trained on
            img_shape = (224, 224, 3)
            img = cv2.resize(image, (img_shape[0], img_shape[1]), interpolation=cv2.INTER_AREA)
            return img
        else:
            # Resize image to what the model is trained on
            img_shape = (320, 320, 3)
            img = cv2.resize(image, (img_shape[0], img_shape[1]), interpolation=cv2.INTER_AREA)

            # Npt using the pre-processing done before training
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            # img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
            # img_clahe_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            #
            # img_clahe_gamma_output = self.adjust_gamma(img_clahe_output, gamma=0.85)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

    def adjust_gamma(image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
