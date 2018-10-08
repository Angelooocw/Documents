
import baxter_interface
import rospy
from sensor_msgs.msg import Image
import cv_bridge
import numpy as np
import cv2
import cv2.cv as cv


rospy.init_node('hola', anonymous = True)

publicador = rospy.Publisher('/robot/xdisplay', Image, latch = True , queue_size = 10)

blank_image3 = np.full((600,1024, 3), 0, dtype = np.uint8)

position = (300, 300)

hola = 'PIPO ES UN CRA'

cv2.putText(blank_image3, hola, position, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255), 4)

msg = cv_bridge.CvBridge().cv2_to_imgmsg(blank_image3)

publicador.publish(msg)

rospy.sleep(1)