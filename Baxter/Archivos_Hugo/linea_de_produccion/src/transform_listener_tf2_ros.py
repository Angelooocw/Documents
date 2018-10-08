#!/usr/bin/env 
# Listener by Hugo 

import rospy
import tf2_ros
import math
import numpy as np
import geometry_msgs.msg 

rospy.init_node("Hola", anonymous = True)

tfBuffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tfBuffer) # Es quien recibe los frames en un buffer por hasta 10 segundos

rate = rospy.Rate(10.0)
rate.sleep()

while not rospy.is_shutdown():
	try:
	    trans = tfBuffer.lookup_transform('torso', 'left_gripper', rospy.Time())
	    break #Para que encuentre la ultima transformacion
	except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
	    rate.sleep()
	    continue

print trans 

#listener = tf.TransformListener()
#(trans,rot) = listener.lookupTransform('/world', '/left_gripper', rospy.Time(0))

#print trans