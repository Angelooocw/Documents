#!/usr/bin/env 

import rospy
from baxter_core_msgs.msg import EndpointState

rospy.init_node("holaasdf")

def callback(msg):
	global x
	x = msg.pose.position

rospy.Subscriber('/robot/limb/left/endpoint_state', EndpointState, callback, queue_size = 1,tcp_nodelay=True)

rospy.sleep(1)

print x