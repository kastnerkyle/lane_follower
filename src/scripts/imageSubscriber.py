import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy

class ImageSubscriber(object):
    def __init__(self):
        super(ImageSubscriber, self).__init__()
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV
        cv2.namedWindow('video_window')
        rospy.Subscriber("/camera/image_raw", Image, self.process_image)
        print "Initialize ImageSubscriber"

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
        called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.cv_image = cv2.resize(self.cv_image, (32, 32))