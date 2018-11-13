#!/usr/bin/env python

import rospy
import baxter_interface
import roslib
import cv2
#import cv2.cv as cv
import cv_bridge
from sensor_msgs.msg import Image
import numpy as np 


#Iniciar Nodo
rospy.init_node('camarita2', anonymous = True)

camara = baxter_interface.CameraController('left_hand_camera')

camara.open()

camara.resolution          = camara.MODES[0]
#camara.exposure            = -1             # range, 0-100 auto = -1
#camara.gain                = -1             # range, 0-79 auto = -1
#camara.white_balance_blue  = -1             # range 0-4095, auto = -1
#camara.white_balance_green = -1             # range 0-4095, auto = -1
#camara.white_balance_red   = -1             # range 0-4095, auto = -1
# camera parametecrs (NB. other parameters in open_camera)
cam_calib    = 0.0025                     # meters per pixel at 1 meter
cam_x_offset = 0.0                       # camera gripper offset
cam_y_offset = 0.0
width        = 960 #640 960                       # Camera resolution
height       = 600 #400 600

foto = None

def open_camera(camera, x_res, y_res):
	print "3 open_camera"
	#called by 1#
        if camera == "left":
            cam = baxter_interface.camera.CameraController("left_hand_camera")
        elif camera == "right":
            cam = baxter_interface.camera.CameraController("right_hand_camera")
        elif camera == "head":
            cam = baxter_interface.camera.CameraController("head_camera")
        else:
            sys.exit("ERROR - open_camera - Invalid camera")

        # close camera
        #cam.close()

        # set camera parameters
        cam.resolution          = (int(x_res), int(y_res))
        cam.exposure            = -1             # range, 0-100 auto = -1
        cam.gain                = -1             # range, 0-79 auto = -1
        cam.white_balance_blue  = -1             # range 0-4095, auto = -1
        cam.white_balance_green = -1             # range 0-4095, auto = -1
        cam.white_balance_red   = -1             # range 0-4095, auto = -1

        # open camera
        cam.open()
	#'back to 1'

open_camera("left", width, height)

def callback(msg):
    # Transforma el mensaje a imagen
    global foto
    foto = cv_bridge.CvBridge().imgmsg_to_cv2(msg) #, "bgr8") #bgr8

    # Abre una ventana con la imagen. 'Foto Nueva' es el nombre de la ventana.

rospy.Subscriber('/cameras/left_hand_camera/image', Image , callback)

img_counter = 0

while not rospy.is_shutdown():
#Capturar un frame
    while np.all(foto) == None:
        #print "hola"
        continue 

    frame = foto
    
    #cv2.drawRectangle()

    #h = height
    #w = width/2
    #resize = cv2.resize(frame, (h,w))
    #Mostrar la imagen
    cv2.imshow('Imagen', frame)

    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
 
cv2.destroyAllWindows()










