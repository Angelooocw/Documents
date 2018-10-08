#!/usr/bin/env 
# Listener by Hugo 

import rospy
import tf
import math
import numpy as np
import geometry_msgs.msg 

rospy.init_node("Hola", anonymous = True)

listener = tf.TransformListener()

'''
#FORMA 1
while not rospy.is_shutdown():
    try:
        (trans,rot) = listener.lookupTransform('/torso', '/left_gripper', rospy.Time(0))
        break # Para imprimir solo la ultima trans
    except tf.Exception:
        continue
'''

#FORMA 2
listener.waitForTransform('/torso', '/left_gripper', rospy.Time(0), rospy.Duration(6))
(trans,rot) = listener.lookupTransform('/torso', '/left_gripper', rospy.Time(0))


print trans
print rot

#print trans