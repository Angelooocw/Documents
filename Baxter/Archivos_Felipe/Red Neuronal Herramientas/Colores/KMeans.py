#!/usr/bin/env python
# -*- coding: cp1252 -*-
import rospy
import cv2,cv_bridge,cv
from sensor_msgs.msg import Image
import baxter_interface
import math as mt
import numpy as np
from traeImagen import *
from baxter_core_msgs.srv import *
import std_srvs.srv
from baxter_core_msgs.msg import CameraSettings,EndpointState,NavigatorState
import operator
from MueveKMeans import *

x_ini=None
y_ini=None
z_ini=None    
button1=None
button0=None
button2=None

def on_state(msg):  
	global button1,button0,button2
	button1=msg.buttons[1]
	button0=msg.buttons[0]
	button2=msg.buttons[2]
      
def endpoint_callback(msg):
	global x_ini,y_ini,z_ini
	x_ini=msg.pose.position.x
	y_ini=msg.pose.position.y
	z_ini=msg.pose.position.z

img=Imagen()

def camera_callback(data, camera_name):
        # Convert image from a ROS image message to a CV image
	global img
        try:
            img.setImg(cv_bridge.CvBridge().imgmsg_to_cv2(data))
        except cv_bridge.CvBridgeError, e:
            print e

        # 3ms wait
        cv.WaitKey(3)

# left camera call back function
def left_camera_callback(data):
        camera_callback(data, "Left Hand Camera")

# right camera call back function
def right_camera_callback(data):
        camera_callback(data, "Right Hand Camera")

# head camera call back function
def head_camera_callback(data):
        camera_callback(data, "Head Camera")

def subscribe_to_camera(camera):
        if camera == "left":
            camera_str = "/cameras/left_hand_camera/image"
	    camera_sub = rospy.Subscriber(camera_str, Image, left_camera_callback)	
        elif camera == "right":
            camera_str = "/cameras/right_hand_camera/image"
	    camera_sub = rospy.Subscriber(camera_str, Image, right_camera_callback)
        elif camera == "head":
            camera_str = "/cameras/head_camera/image"
	    camera_sub = rospy.Subscriber(camera_str, Image, head_camera_callback)
        else:
            sys.exit("ERROR - subscribe_to_camera - Invalid camera")

def open_camera(camera, x_res, y_res):
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
        cam.resolution          = int(x_res), int(y_res)
        cam.exposure            = -1             # range, 0-100 auto = -1
        cam.gain                = -1             # range, 0-79 auto = -1
        cam.white_balance_blue  = -1             # range 0-4095, auto = -1
        cam.white_balance_green = -1             # range 0-4095, auto = -1
        cam.white_balance_red   = -1             # range 0-4095, auto = -1

        # open camera
        cam.open()

def reset_cameras():
        reset_srv = rospy.ServiceProxy('cameras/reset', std_srvs.srv.Empty)
        rospy.wait_for_service('cameras/reset', timeout=10)
        reset_srv()

def close_camera(camera):
        if camera == "left":
            cam = baxter_interface.camera.CameraController("left_hand_camera")
        elif camera == "right":
            cam = baxter_interface.camera.CameraController("right_hand_camera")
        elif camera == "head":
            cam = baxter_interface.camera.CameraController("head_camera")
        else:
            sys.exit("ERROR - close_camera - Invalid camera")
        # close camera
        cam.close()

def CentroMasa(I,Color): 

	Masa=0
	sumax=0
	sumay=0
	cord=[0,0]
	for i in range(I.shape[0]): 
		for j in range(I.shape[1]):
			pixel = I[i][j]
			sumax += pixel*(j + 1)
			sumay += pixel*(i + 1)
			Masa += pixel
		
	

	cord[0] = sumax / Masa
	cord[1] = sumay / Masa
	cord[1] = ((float)(Color.shape[0]) / (float)(I.shape[0]))*cord[1]
	cord[0] = ((float)(Color.shape[1]) / (float)(I.shape[1]))*cord[0]
	return cord

def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.cv.BoxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]

    return img_crop

def math_calc_dist(p1,p2):
    return mt.sqrt(mt.pow((p2[0] - p1[0]),2) + mt.pow((p2[1] - p1[1]),2))
def main():
	Posiciones=[] # Guardo las posiciones iniciales de configuracion
	featuringRead=np.loadtxt('featuring.out',delimiter=' ')
	featuringRead=np.float32(featuringRead)
	# Apply KMeans
	compactness,labels,centers = cv2.kmeans(featuringRead,K=3,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,20,0),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
	A=featuringRead[labels.ravel()==0]
	B=featuringRead[labels.ravel()==1]
	C=featuringRead[labels.ravel()==2]
	MIN_OBJECT_AREA=20*20
	MAX_OBJECT_AREA=40*40
	Multi=cv2.imread("multi700x490.jpg",1)
	Multihsv=cv2.cvtColor(Multi,cv2.COLOR_BGR2HSV)
	Multihue = np.zeros(Multihsv.shape, dtype=np.uint8)
	cv2.mixChannels([Multihsv],[Multihue],[0,0])
	rospy.init_node("Analisis_de_Color")
	"""reset_cameras()
	close_camera("left")
        close_camera("right")
        close_camera("head")"""
	open_camera("right",960,600)
	subscribe_to_camera("right")
	#screen_pub = rospy.Publisher('/robot/xdisplay', Image, queue_size=10)
	endpoint_sub = rospy.Subscriber('/robot/limb/right/endpoint_state',EndpointState,callback=endpoint_callback) 
	state_sub = rospy.Subscriber('robot/navigators/right_navigator/state',NavigatorState,callback=on_state)
	#rate = rospy.Rate(3)
	print 'Ahora graba posiciones'
	# Posiciones
	########3########
	#################
	#2######0#######1
	#################
	########4########
	#
	while 1:
		if button0:
			Posiciones.append([x_ini,y_ini])
			while button0:
				continue
		if button1 or button2:
			break
	print 'posiciones grabadas'
	blank_image = np.zeros((600,960,3), np.uint8)
	for i in range(20):
		cv2.line(blank_image,(int((960/20)*(i+1)),0),(int((960/ 20)*(i + 1)),600),(255,0,0),5)
		cv2.line(blank_image,(0,int(600/ 20)*(i + 1)),(960,int((600/20)*(i + 1))),(255,0,0),5)
	cv2.circle(blank_image,(int(centers[0][0]),int(centers[0][1])),20,(0,255,0),-1)
	cv2.circle(blank_image,(int(centers[1][0]),int(centers[1][1])),20,(0,255,0),-1)
	cv2.circle(blank_image,(int(centers[2][0]),int(centers[2][1])),20,(0,255,0),-1)
	for i in range(len(A)):
		cv2.circle(blank_image,(int(A[i][0]),int(A[i][1])),4,(0,255,255),-1)
	for i in range(len(B)):
		cv2.circle(blank_image,(int(B[i][0]),int(B[i][1])),4,(0,255,255),-1)
	for i in range(len(C)):
		cv2.circle(blank_image,(int(C[i][0]),int(C[i][1])),4,(0,255,255),-1)
	Muevee = Mueve('right',Posiciones) 
	erodeElement = cv2.getStructuringElement( cv2.MORPH_RECT,(3,3)) # selecciono el tipo de kernel
	while 1:
		Etiquetas=[] # Guardo las etiquetas de KMeans
		Angulo=[]
		Dimensiones=[]
		Coordenadas=[]
		SetFrames=[]
		Grays=[]
		if img.getImg() is None:
			continue
		while len(SetFrames)<20:
			SetFrames.append(np.copy(img.getImg()))
			Grays.append(0)
		Color = img.getImg()
		gray=cv2.cvtColor(Color,cv2.COLOR_BGR2GRAY)
		gray=cv2.equalizeHist(gray)
		gray=cv2.GaussianBlur(gray,(5, 5), 0, 0)
		"""sobel64f = cv2.Sobel(gray,cv2.CV_64F,1,1,ksize=5)
   		abs_sobel64f = np.absolute(sobel64f)
   		sobel_8u = np.uint8(abs_sobel64f)
		sobel_8u = cv2.erode(sobel_8u,erodeElement)"""
		gray = cv2.Canny(gray, 100, 150, 3)
		"""for m in range(gray.shape[:2][0]):
				for n in range(gray.shape[:2][1]):
					if gray[m][n]==0:
						gray[m][n]=sobel_8u[m][n]"""
		for i in range(len(SetFrames)):
			Grays[i]=cv2.cvtColor(SetFrames[i],cv2.COLOR_BGR2GRAY)
			Grays[i]=cv2.equalizeHist(Grays[i])
			Grays[i]=cv2.GaussianBlur(Grays[i],(5, 5), 0, 0)
			Grays[i]=cv2.Canny(Grays[i], 100, 150, 3)
		for i in range(len(Grays)):
			gray=cv2.bitwise_or(gray,Grays[i])
		cv2.imshow("gray",gray)
		(contours,hierarchy)=cv2.findContours(gray.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		if len(hierarchy)>0:
		        index=0
		        while index != -1:
				moment = cv2.moments(contours[index])
				area = moment['m00']
				if area>MIN_OBJECT_AREA and area < MAX_OBJECT_AREA:
					rect = cv2.minAreaRect(contours[index])
					roi = crop_minAreaRect(Color,rect)
					if roi.shape[:2][1]>0 and roi.shape[:2][0]>0:
						hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
						hue = np.zeros(hsv.shape, dtype=np.uint8)
						cv2.mixChannels([hsv],[hue], [0,0])
						hist = cv2.calcHist([hue],[0],None,[180],[0,180])
						cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
						backproj=cv2.calcBackProject([Multihue],[0],hist,[0,180],1)
						ret,backproj=cv2.threshold(backproj,127,255,cv2.THRESH_BINARY)
						receiver = CentroMasa(backproj,Color)
						dist=[math_calc_dist(receiver,centers[0]),math_calc_dist(receiver,centers[1]),math_calc_dist(receiver,centers[2])]
						min_index,min_value = min(enumerate(dist),key=operator.itemgetter(1))
						#print 'centro: ',centers[min_index],' distancia: ',dist[min_index]
						Coordenadas.append(list([int(moment['m10']/area),int(moment['m01']/area)]))
						Etiquetas.append(min_index)
						Angulo.append(rect[2])
						Dimensiones.append(rect[1])
						box = cv2.cv.BoxPoints(rect)
	     					box = np.int0(box)
						cv2.circle(blank_image,(int(receiver[0]),int(receiver[1])),5,(0,0,255),-1)
						cv2.drawContours(Color,[box],0,(0,0,255),2)
					

				index=hierarchy[0][index][0]

		cv2.imshow("Imagen Filtrada",Color)
		cv2.imshow("Enrejado",blank_image)
		#msgsub = cv_bridge.CvBridge().cv2_to_imgmsg(blank_image, encoding="8UC3")
		#screen_pub.publish(msgsub)
		Color1=[]
		Color2=[]
		Color3=[]
		print Etiquetas
		for i in range(len(Coordenadas)):
			if Etiquetas[i]==0:
				Color1.append(i)
			elif Etiquetas[i]==1:
				Color2.append(i)
			else:
				Color3.append(i)
		for i in Color1:
			Muevee.CA(Coordenadas[i][0],Coordenadas[i][1],Etiquetas[i],Angulo[i],Dimensiones[i])
		for i in Color2:
			Muevee.CA(Coordenadas[i][0],Coordenadas[i][1],Etiquetas[i],Angulo[i],Dimensiones[i])
		for i in Color3:
			Muevee.CA(Coordenadas[i][0],Coordenadas[i][1],Etiquetas[i],Angulo[i],Dimensiones[i])
		Muevee.randomm()
		Muevee.mover_baxter('base',Muevee.pose[:3],Muevee.pose[3:6])
		if cv2.waitKey(1) & 0xFF == ord('q'): # Indicamos que al pulsar "q" el programa se cierre
		   break
	#rate.sleep()
	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


