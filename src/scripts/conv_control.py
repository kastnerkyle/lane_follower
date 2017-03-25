#!/usr/bin/env python

#import modules
from utils import *

import numpy as np
import cv2
import rospy
from cmdVelPublisher import CmdVelPublisher
from imageSubscriber import ImageSubscriber



class RobotControllerConv(CmdVelPublisher, ImageSubscriber, object):
    """ This script helps find colors masks """

    def __init__(self):
        rospy.init_node('robot_control_convnet')
        super(RobotControllerConv, self).__init__()
        self.model = self.get_model()
        self.model.load_weights('binary_86.h5')



    def get_model(self):
        model = Sequential([
            Convolution2D(32,3,3, border_mode='same', activation='relu', input_shape=(16, 32, 1)),
            MaxPooling2D(),
            Convolution2D(64,3,3, border_mode='same', activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')
            ])
        model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def robot_control(self, action):
        # action:
        # 0 = forward
        # 1 = leftTurn
        # 2 = rightTurn
        # 3 = stop

        try:
            if action < 0 or action > 3:
                raise ValueError("Action is invalid")
            self.state[action].__call__()
        except:
            # make robot stop
            print "Invalid action - stopping robot"
            self.state[3].__call__()

        self.sendMessage()
        # rospy.sleep(.3) # use desired actioni for one second
        # self.state[3].__call__() # set robot to stop
        self.sendMessage()

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.cv_image is None:
                input_image = self.binary_image[16:,:]
                cv2.imshow('video_window', input_image)
                cv2.waitKey(5)
                a = np.argmax(self.model.predict((input_image / 255.0).reshape([1, 16, 32, 1])))
                self.robot_control(a)
                print a
            r.sleep()

if __name__ == '__main__':
    node = RobotControllerConv()
    node.run()
