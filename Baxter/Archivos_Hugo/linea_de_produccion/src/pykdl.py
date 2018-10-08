#!/usr/bin/env 

import rospy
from baxter_pykdl import baxter_kinematics
import numpy as np
import math
import tf
import baxter_interface

rospy.init_node('nodo_de_prueba',anonymous=True)

izq = baxter_kinematics('left')

#print "forward"
#print izq.forward_position_kinematics()


print 'qua from eul'
rot = tf.transformations.quaternion_from_euler(math.pi,0,0)
trans = [0.3, 0.1, 0.2]
#trans = [0.582583, -0.180819, 0.216003]
print rot
a = rot.tolist()
print a 
print '#################'

print "inverse_kinematics"
angulos = izq.inverse_kinematics(trans)
print "angulos en grados"
print angulos


angulos = np.radians(angulos)
print "angulos en radianes"
print angulos


###########################

arm = baxter_interface.Limb('left')
print arm.joint_names()

names = ['left_s0', 'left_s1', 'left_w0', 'left_w1', 'left_w2', 'left_e0', 'left_e1']

movimiento =  dict(zip(names,angulos))

print "movimientoooo"
print movimiento

arm.move_to_joint_positions(movimiento)

print arm.joint_angles()
#arm.set_joint_positions(movimiento)

#print derecha.joints_to_kdl('positions')
