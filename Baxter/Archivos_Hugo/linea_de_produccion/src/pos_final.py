#!/usr/bin/env 
#by Hugo Ubilla

import rospy
import baxter_interface


rospy.init_node("anonimo", anonymous = True)

brazo_izq =  baxter_interface.Limb('left')

print brazo_izq.endpoint_pose()
