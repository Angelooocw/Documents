#!/usr/bin/env python
# -*- coding: cp1252 -*-
import rospy
import cv2,cv_bridge,cv
from sensor_msgs.msg import Image
import baxter_interface
from traeImagen import *
from DNNGood import *
from MueveGood import *
import math as mt
import numpy as np
from baxter_core_msgs.srv import *
import std_srvs.srv
from baxter_core_msgs.msg import CameraSettings,EndpointState,NavigatorState

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


def EcualizacionUp(contours,OptimalDescriptor):
        normalizacionUp=[]
	PtosContorno = len(contours) #Cantidad de puntos que forman el contorno
	Complejo=[0.0,0.0]
	for i in range(OptimalDescriptor):
		index = 1.0*i*float(PtosContorno)/float(OptimalDescriptor)
		j = int(index)
		k = index - j
		if (j == PtosContorno - 1):
			Complejo[0] = contours[j][0] #Real
			Complejo[1] = contours[j][1] #Imaginaria
		
		else:
			Complejo[0] = contours[j][0]*(1-k) + contours[j+1][0]*(k) #Real
			Complejo[1] = contours[j][1]*(1-k) + contours[j+1][1]*(k) #Imaginaria
		
		normalizacionUp.append(Complejo[:])
	
	return normalizacionUp

def EcualizacionDown(contours,OptimalDescriptor):
	normalizacionDown=[]
	Inicial=[0.0,0.0]
	for i in range(OptimalDescriptor):
		normalizacionDown.append(Inicial[:])
	PtosContorno = len(contours) #Cantidad de puntos que forman el contorno
	for i in range(PtosContorno):
		normalizacionDown[int(i*float(OptimalDescriptor)/float(PtosContorno))][0] += contours[i][0]
		normalizacionDown[int(i*float(OptimalDescriptor)/float(PtosContorno))][1] += contours[i][1]
	return normalizacionDown

def EcualizacionDescriptor(contours):
	PtosContorno = len(contours) #Cantidad de puntos que forman el contorno
	OptimalDescriptor = 30 #cantidad de descriptores optimos para forma
	if (OptimalDescriptor > PtosContorno):
		normalizacion = EcualizacionUp(contours, OptimalDescriptor) #Interpolacion
	else:
		normalizacion = EcualizacionDown(contours, OptimalDescriptor) #Suma vectorial	
	return normalizacion

def Rotacion(contours,m):
	Rotacion=[]
	PtosContorno = len(contours) #Cantidad de puntos que forman el contorno
	for i in range(m,PtosContorno):
		Rotacion.append(contours[i])
	for i in range(m):
		Rotacion.append(contours[i])
	return Rotacion


def NormaContorno(contours):
	Norma = 0.0
	PtosContorno = len(contours) #Cantidad de puntos que forman el contorno
	for i in range(PtosContorno):
		Norma += pow(contours[i][0],2) + pow(contours[i][1], 2)
	Norma = mt.sqrt(Norma)
	return Norma


def NormaComplejo(contours):
	Norma = 0.0
	Norma += pow(contours[0], 2) + pow(contours[1], 2)
	Norma = mt.sqrt(Norma)
	return Norma

def NSP(contoursA,contoursB):
	NSP=[0.0,0.0]
	Norma = NormaContorno(contoursA)*NormaContorno(contoursB)
	PtosContorno = len(contoursA) #Cantidad de puntos que forman el contorno
	for i in range(PtosContorno):
		NSP[0] += (contoursA[i][0]*contoursB[i][0] + contoursA[i][1]*contoursB[i][1])
		NSP[1] += (contoursA[i][1]*contoursB[i][0] - contoursA[i][0]*contoursB[i][1])
	NSP[0] = NSP[0]/Norma
	NSP[1] = NSP[1]/Norma
	return NSP

def ACRNorma(contours):
	PtosContorno = len(contours) #Cantidad de puntos que forman el contorno
	Patron=[]
	maxi = 0.0
	Normas=[]
	for i in range(PtosContorno + 1):
		Rotacionn = Rotacion(contours,i)[:]
		Norma = NormaContorno(contours)*NormaContorno(Rotacionn)
		Patron.append(NSP(contours,Rotacionn)[:])
		Patron[i][0] = Patron[i][0]*Norma #Desnormalizar NSP
		Patron[i][1] = Patron[i][1]*Norma #Desnormalizar NSP
	
	for i in range(len(Patron)):
		if (mt.sqrt(pow(Patron[i][0], 2) + pow(Patron[i][1], 2)) > maxi):
			maxi = mt.sqrt(pow(Patron[i][0], 2) + pow(Patron[i][1], 2))
		
	
	for i in range(len(Patron)):
		Normas.append(mt.sqrt(pow(Patron[i][0], 2) + pow(Patron[i][1], 2)) / maxi)
	return Normas


def ACRContour(contours):
	PtosContorno = len(contours) #Cantidad de puntos que forman el contorno
	Patron=[]
	maxi = 0.0
	for i in range(PtosContorno + 1):
		Rotacionn = Rotacion(contours, i)[:]
		Norma = NormaContorno(contours)*NormaContorno(Rotacionn)
		Patron.append(NSP(contours, Rotacionn)[:])
		Patron[i][0] = Patron[i][0]*Norma #Desnormalizar NSP
		Patron[i][1] = Patron[i][1]*Norma #Desnormalizar NSP
	
	for i in range(len(Patron)):
		if (mt.sqrt(pow(Patron[i][0], 2) + pow(Patron[i][1], 2)) > maxi):
			maxi = mt.sqrt(pow(Patron[i][0], 2) + pow(Patron[i][1], 2))
	
	for i in range(len(Patron)):
		Patron[i][0] = Patron[i][0] / maxi
		Patron[i][1] = Patron[i][1] / maxi
	return Patron

def ICRNorma(contoursA,contoursB):
	PtosContorno = len(contoursA) #Cantidad de puntos que forman el contorno
	Patron=[]
	Normas=[]
	maximo = 0.0
	for i in range(PtosContorno):
		Rotacionn = Rotacion(contoursB, i)[:]
		Patron.append(NSP(contoursA, Rotacionn)[:])
		Norma = NormaContorno(contoursA)*NormaContorno(Rotacionn)
		Patron[i][0] = Patron[i][0]*Norma
		Patron[i][1] = Patron[i][1]*Norma
	for i in range(PtosContorno):
		Normas.append(mt.sqrt(pow(Patron[i][0],2) + pow(Patron[i][1],2)))
		maximo = max(Normas[i], maximo)
	return maximo

def ICRComplejo(contoursA,contoursB):
	PtosContorno = len(contoursA) #Cantidad de puntos que forman el contorno
	Patron=[]
	Normas=[]
	maximo = 0.0
	for i in range(PtosContorno):
		Rotacionn = Rotacion(contoursB, i)[:]
		Patron.append(NSP(contoursA, Rotacionn)[:])
		Norma = NormaContorno(contoursA)*NormaContorno(Rotacionn)
		Patron[i][0] = Patron[i][0]*Norma
		Patron[i][1] = Patron[i][1]*Norma
	for i in range(PtosContorno):
		Normas.append(mt.sqrt(pow(Patron[i][0], 2) + pow(Patron[i][1], 2)))
		maximo = max(Normas.at(i), maximo)
	for i in range(PtosContorno):
		if (Normas[i] == maximo): 
			indice = i
			break

	return Patron[indice]

def Codificacion(contours):
	codificacion=[]
	con=[]
	CAPoint=[0.0,0.0]
	con.append(contours[0])
	for i in range(len(contours)-1,0,-1):
		con.append(contours[i])
	for i in range(1,len(contours)):
		CAPoint[0] = (con[i][0][0] - con[i-1][0][0])
		CAPoint[1] = (con[i][0][1] - con[i-1][0][1])
		codificacion.append(CAPoint[:])
	CAPoint[0] = (con[0][0][0] - con[len(con)-1][0][0])
	CAPoint[1] = (con[0][0][1] - con[len(con)-1][0][1])
	codificacion.append(CAPoint[:])
	return codificacion

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

def main():

	Plantillas=[] # Guardo el nombre de las plantillas
	Objetos=[] # Guardo las imagenes de las plantillas
	EcualizacionFourier=[] # Contorno ecualizado
	Posiciones=[] # Guardo las posiciones iniciales de configuracion
	with open("Plantillasaux.txt","r") as f:
		for line in f:
			Plantillas.append(line)
			Plantillas[-1]=Plantillas[-1].rstrip("\r\n")
	for i in range(len(Plantillas)):
		Objetos.append(cv2.imread(Plantillas[i], 1))
	for z in range(len(Objetos)):
		OBJETOSgray=cv2.cvtColor(Objetos[z],cv2.COLOR_BGR2GRAY)
		OBJETOSgray=cv2.equalizeHist(OBJETOSgray)
		OBJETOSgray=cv2.GaussianBlur(OBJETOSgray,(5, 5), 0, 0)
		OBJETOSgray=cv2.Canny(OBJETOSgray, 100, 150, 3)
		(con,hie)=cv2.findContours(OBJETOSgray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		if hie is not None:
                        index=0
                        while index != -1:
                                moment = cv2.moments(con[index])
                                area = moment['m00']
				if area > 10 * 10:
					codificacion = Codificacion(con[index])[:]
					codificacion = EcualizacionDescriptor(codificacion)[:]
					Patron = ACRNorma(codificacion)[:]
					EcualizacionFourier.append(codificacion[:])
					"""while 1:
						cv2.drawContours(Objetos[z],con,index,np.array([0, 255, 0], dtype=np.uint8).tolist(),3,8)
						cv2.imshow("Formas", Objetos[z])
						if cv2.waitKey(0) & 0xFF == ord('q'): # Indicamos que al pulsar "q" el programa se cierre
                        				break"""
				index=hie[0][index][0]

	cv2.destroyAllWindows()
	Nombres_Plantillas=["Alicate","Destornillador","Martillo","Taladro","Llave"]
	MIN_OBJECT_AREA = 20*20
	rospy.init_node("Analisis_de_Contornos")
	reset_cameras()
	close_camera("left")
        close_camera("right")
        close_camera("head")
	open_camera("left", 1280, 800)
	subscribe_to_camera("left")
	endpoint_sub = rospy.Subscriber('/robot/limb/left/endpoint_state',EndpointState,callback=endpoint_callback) 
	state_sub = rospy.Subscriber('robot/navigators/left_navigator/state',NavigatorState,callback=on_state)
	screen_pub = rospy.Publisher('/robot/xdisplay', Image, queue_size=10)
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
	Muevee = Mueve('left',Posiciones)  # En este punto debo pasar todas las posiciones iniciales
	DNNH=DNN()
	DNNH.create_graph()
	with DNNH.g.as_default() as g:
		with tf.Session() as sess:
			DNNH.saver.restore(sess, "./my_model_final_good2.ckpt")
			while 1:
				Coordenadas=[]
				Etiquetas=[]
				Angulo=[]
				Dimensiones=[]
				SetFrames=[]
				Grays=[]
				if img.getImg() is None:
					continue
				while len(SetFrames)<20:
					SetFrames.append(np.copy(img.getImg()))
					Grays.append(0)
				CA = img.getImg()
				gray=cv2.cvtColor(CA,cv2.COLOR_BGR2GRAY)
				gray=cv2.equalizeHist(gray)
				gray=cv2.GaussianBlur(gray,(5, 5), 0, 0)
				gray=cv2.Canny(gray, 100, 150, 3)
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
						features = []
						ACFS=[]
						ICFS=[]
						moment = cv2.moments(contours[index])
				                area = moment['m00']	
						rect = cv2.minAreaRect(contours[index])
						box = cv2.cv.BoxPoints(rect)
     						box = np.int0(box)
						if area>MIN_OBJECT_AREA:
							codificacion = Codificacion(contours[index])[:]
							codificacion = EcualizacionDescriptor(codificacion)[:]
							Patron = ACRNorma(codificacion)[:]
							for i in range(len(EcualizacionFourier)):
								ICF=ICRNorma(EcualizacionFourier[i],codificacion)/(NormaContorno(EcualizacionFourier[i])*NormaContorno(codificacion)) #ICF
								ICFS.append(ICF)
							if(max(ICFS)>=0.84):
								for i in range(len(EcualizacionFourier)):
									ACR=NormaComplejo(NSP(ACRContour(EcualizacionFourier[i]),ACRContour(codificacion))[:]) #ACR
									ACFS.append(ACR)
									features.append(ACFS[i]) 
									features.append(ICFS[i])
								#if(max(ACFS)>=0.95):
								# Usar la DNN que ya esta entrenada
								clase=int(DNNH.Entrada(features))
								x,y,w,h = cv2.boundingRect(contours[index])
								size,baseline = cv2.getTextSize(Nombres_Plantillas[clase], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1) 
							     	x1=x + ((w - size[0]) / 2) 
							     	y1=y + ((h + size[1]) / 2) + baseline
							     	x2=x + ((w - size[0]) / 2) + size[0]
							     	y2=y + ((h + size[1]) / 2) - size[1]
								cv2.rectangle(CA, (x, y), (x+w, y+h),np.array([0, 255, 0], dtype=np.uint8).tolist(),4)
								cv2.putText(CA,Nombres_Plantillas[clase],(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.4,np.array([0,255,0],dtype=np.uint8).tolist(),1,8)
								cv2.drawContours(CA,[box],0,(0,0,255),2)
								Coordenadas.append(list([int(moment['m10']/area),int(moment['m01']/area)]))
								Etiquetas.append(Nombres_Plantillas[clase])
								Angulo.append(rect[2])
								Dimensiones.append(rect[1])
								
									
						index=hierarchy[0][index][0]

				cv2.imshow("Imagen Filtrada",CA)
				msgsub = cv_bridge.CvBridge().cv2_to_imgmsg(CA, encoding="bgra8")
				screen_pub.publish(msgsub)
				Martillos=[]
				Alicates=[]
				Detornilladores=[]
				Taladros=[]
				Llaves=[]
				for i in range(len(Coordenadas)):
					if Etiquetas[i]=='Martillo':
						Martillos.append(i)
					elif Etiquetas[i]=='Alicate':
						Alicates.append(i)
					elif Etiquetas[i]=='Destornillador':
						Detornilladores.append(i)
					elif Etiquetas[i]=='Taladro':
						Taladros.append(i)
					else:
						Llaves.append(i)
				for i in Martillos:
					Muevee.CA(Coordenadas[i][0],Coordenadas[i][1],Etiquetas[i],Angulo[i],Dimensiones[i])
				for i in Alicates:
					Muevee.CA(Coordenadas[i][0],Coordenadas[i][1],Etiquetas[i],Angulo[i],Dimensiones[i])
				for i in Detornilladores:
					Muevee.CA(Coordenadas[i][0],Coordenadas[i][1],Etiquetas[i],Angulo[i],Dimensiones[i])
				for i in Taladros:
					Muevee.CA(Coordenadas[i][0],Coordenadas[i][1],Etiquetas[i],Angulo[i],Dimensiones[i])
				for i in Llaves:		
					Muevee.CA(Coordenadas[i][0],Coordenadas[i][1],Etiquetas[i],Angulo[i],Dimensiones[i])
				Muevee.randomm()
				Muevee.mover_baxter('base',Muevee.pose[:3],Muevee.pose[3:6])
				if cv2.waitKey(1) & 0xFF == ord('q'): # Indicamos que al pulsar "q" el programa se cierre
				   break
			#rate.sleep()
			cv2.destroyAllWindows()









if __name__ == "__main__":
    main()

	
