from styx_msgs.msg import TrafficLight
import cv2
from keras.models import load_model
import rospy
import tensorflow as tf
from keras import backend as K

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        K.clear_session()
        self.model = load_model('squeezenet_model_75_0005.h5')
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        rospy.logwarn('Classifying image ')
        # Resize image to what the model is trained on
        img_shape = (224, 224, 3)
        img = cv2.resize(image, (img_shape[0], img_shape[1]), interpolation=cv2.INTER_AREA)

        # Get the prediction for the image
        with self.graph.as_default():
            rospy.logwarn('Predicting image ')
            preds = self.model.predict(img[None,:,:,:], verbose=1)
            rospy.logwarn('Checking pred result ')
            state = preds[0].argmax()
            prob = preds[0][state]

            rospy.logwarn('prediction ' + str(prob))
            rospy.logwarn('state ' + str(state))
            rospy.logwarn('All preds ' + str(preds))

            if state == 1:
                return TrafficLight.RED
            else:
                return TrafficLight.UNKNOWN