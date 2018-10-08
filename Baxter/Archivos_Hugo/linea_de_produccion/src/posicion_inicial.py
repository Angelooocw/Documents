#!/usr/bin/env 

import baxter_interface
import rospy
import tf
import numpy 

rospy.init_node("pos_ini")

#boton.posicion_inicial()
brazo = baxter_interface.Limb('left')
print brazo.endpoint_pose()

'''x = brazo.endpoint_pose()['position'][0]
y = brazo.endpoint_pose()['position'][1]
z = brazo.endpoint_pose()['position'][2]

print x,y,z'''


print tf.transformations.euler_from_quaternion(brazo.endpoint_pose()['orientation'])