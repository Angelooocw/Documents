#!/usr/bin/env 

import rospy
import baxter_interface

rospy.init_node("nuevo")

gripper_izq = baxter_interface.Gripper('left')

#gripper_izq.calibrate()

print gripper_izq.parameters()

print gripper_izq.gripping()

print gripper_izq.force()