#!/usr/bin/env python
import numpy as np
import math 
import cv2
import rospy
import baxter_interface
import cv_bridge
import roslib
import tf
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from math import hypot
from grasp_image import grasping,get_points


import random
#from baxterfunctions import *

def prediccion(file):
	#x = load_img(file,target_size=(longitud,altura))
	#x = img_to_array(x)
	#x = np.expand_dims(x,axis=0)
	y=cv2.cvtColor(file,cv2.COLOR_BGRA2BGR)
	x=cv2.resize(y,(longitud,altura))
	x=img_to_array(x)
	x=np.expand_dims(x,axis=0)
	arreglo=cnn.predict(x) #[[1,0,0,0,0]]
	#print arreglo
	resultado=arreglo[0]
	respuesta=np.argmax(resultado)

	#print resultado

	if respuesta==0:
		herramienta='Alicate'
	elif respuesta==1:
		herramienta='Calculadora'
	elif respuesta==2:
		herramienta='Cuchillo'
	elif respuesta==3:
		herramienta='Destornillador'
	elif respuesta==4:
		herramienta='Gamepad'
	elif respuesta==5:
		herramienta='Llave'
	elif respuesta==6:
		herramienta='Martillo'
	elif respuesta==7:
		herramienta='Mouse'
	elif respuesta==8:
		herramienta='Reloj'
	elif respuesta==9:
		herramienta='Taladro'
	elif respuesta==10:
		herramienta='Telefono'
	return herramienta

def mensaje_matriz_a_pose(T, frame):
	t = PoseStamped()
	t.header.frame_id = frame
	t.header.stamp = rospy.Time.now()
	translacion = tf.transformations.translation_from_matrix(T)
	orientacion = tf.transformations.quaternion_from_matrix(T)
	t.pose.position.x = translacion[0]
	t.pose.position.y = translacion[1]
	t.pose.position.z = translacion[2]
	t.pose.orientation.x = orientacion[0]
	t.pose.orientation.y = orientacion[1]
	t.pose.orientation.z = orientacion[2]
	t.pose.orientation.w = orientacion[3]        
	return t

def mover_baxter( source_frame, trans, rot):

	nombre_servicio = '/ExternalTools/'+ limb +'/PositionKinematicsNode/IKService'
	servicio_ik = rospy.ServiceProxy(nombre_servicio,SolvePositionIK)
	frame = source_frame   

	# Promedio de velocidad del brazo
	limb_interface.set_joint_position_speed(0.5)

	matrix = tf.transformations.concatenate_matrices(tf.transformations.translation_matrix(trans),tf.transformations.euler_matrix(rot[0],rot[1],rot[2]))
        
	rospy.wait_for_service(nombre_servicio,10)
	ik_mensaje = SolvePositionIKRequest()
	ik_mensaje.pose_stamp.append(mensaje_matriz_a_pose(matrix, frame))

	try:
		respuesta = servicio_ik(ik_mensaje)
	except:
		print "Movimiento no ejecutado"

	print respuesta.isValid[0]

	if respuesta.isValid[0] == True:
		movimiento =  dict(zip(respuesta.joints[0].name, respuesta.joints[0].position))
		limb_interface.move_to_joint_positions(movimiento)
	else:
		print "Movimiento no ejecutado"
		print respuesta.joints[0].position
		print respuesta.joints[0].name

def pixel_to_baxter(px, dist):
	x = ((px[1] - (height / 2)) * cam_calibracion * dist)+ pose_i[0] + cam_x_offset
	y = ((px[0] - (width / 2)) * cam_calibracion * dist)+ pose_i[1] + cam_y_offset
	
	return (x, y)

def send_image(image):	#Shows an image on Baxter's screen
	img = cv2.imread(image)	#Reads an image
	msg = cv_bridge.CvBridge().cv2_to_imgmsg(img, encoding="bgr8") #Makes the opencv-ros bridge, converts an image to msg
	pub.publish(msg) #Shows the image on Baxter's head screen
	rospy.sleep(0.1)

def QtoE(): #Quaternion to Euler. Converts Quaternion angles(x, y, z, w) into Euler angles (x, y ,z) and prints them
	euler = tf.transformations.euler_from_quaternion(limb_interface.endpoint_pose()['orientation'])
	print ("Arm positions and Quaternion angles")
	print (limb_interface.endpoint_pose())
	print ("Arm Euler angles: ", euler)

#Detecta Bordes de objetos de la camara y retorna los contornos validos
def img_process(image):
	hh,ww=image.shape[:2]
	roi=image[300:hh, 0:ww]
	gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#gray= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	blur= cv2.GaussianBlur(gray,(5,5),0)
	canny= cv2.Canny(blur,50,200)

	#Morphologic, para completar bordes
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	dilated = cv2.dilate(canny,kernel)

	#cv2.imshow('camara dilatada',dilated)

	(contornos,_) = cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cont_validos=[]
	for c in contornos:
		x,y,w,h=cv2.boundingRect(c)
		area=w*h
		if area<=1000 or y<300:
			continue
		cont_validos.append(c)
		puntos_pre.append((x,y))
	return cont_validos

#Busca contorno en el centro de la imagen captada
def ucontorno(image):
	gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#gray= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	blur= cv2.GaussianBlur(gray,(5,5),0)
	canny= cv2.Canny(blur,50,200)

	#Morphologic, para completar bordes
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	dilated = cv2.dilate(canny,kernel)

	(contornos,_) = cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for c in contornos:
		x,y,w,h=cv2.boundingRect(c)

		area=w*h
		#Se limita la zona para ignorar contornos en los bordes
		if area<=1000 or y<300 or x<400 or x>880:
			continue

		cv2.rectangle(image, (x,y),(x+w,y+h),(255,0,255),2)
		ct.append(c)
	return ct

xi=0
yi=0
#Realiza recortes por cada contorno detectado
def recortes_img(image, contornos):
	recortes=[]
	for c in contornos:
		x,y,w,h=cv2.boundingRect(c)
		xc,yc=x+w/2,y+h/2
		global xi,yi
		yi=y-margen_img
		yf=y+h+margen_img
		xi=x-margen_img
		xf=x+w+margen_img

		if yi<0:
			yi=0
		if xi<0:
			xi=0

		crop=image[yi:yf,xi:xf]
		recortes.append(crop)

	return recortes


#Reescala las imagenes recortadas para luego ser pasadas al predictor
def reescalado(recorte):

	old_size= recorte.shape[:2]
	ratio=float(tamano_deseado)/max(old_size)
	new_size = tuple([int(o*ratio) for o in old_size])
	im = cv2.resize(recorte, (new_size[1], new_size[0]))

	delta_w = tamano_deseado - new_size[1]
	delta_h = tamano_deseado - new_size[0]
	top, bottom = delta_h//2, delta_h-(delta_h//2)
	left, right = delta_w//2, delta_w-(delta_w//2)

	color = [255, 255, 255]
		
	new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)

	return new_im

#Prediccion de Objetos
def prediccion_obj(recortes):
	nombres_obj=[]
	for i in recortes:
		img=reescalado(i)
		nombres_obj.append(prediccion(img))
	return nombres_obj


#Dibuja sobre la imagen con respecto a los objetos y calcula angulo de inclinacion y retorna punto de agarre
def info_and_angles(img,contornos,nombres):
	j=0
	for c in contornos:
		x,y,w,h=cv2.boundingRect(c)
		cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
		cv2.putText(img, nombres[j],(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,np.array([0,0,0],dtype=np.uint8).tolist(),1,8)

		#Rectangulo de area minima
		rect=cv2.minAreaRect(c)
		box=cv2.cv.BoxPoints(rect)
		box=np.int0(box)
		cv2.drawContours(img,[box],0,(0,0,255),2)

		##Puntos rectangulo minimo, se toma como primer punto el que este mas abajo en el eje y
		##luego los siguientes se toman en el sentido de las agujas del reloj
		xb1,yb1=box[0][0],box[0][1]
		xb2,yb2=box[1][0],box[1][1]
		xb3,yb3=box[2][0],box[2][1]

		dist1=hypot(xb2-xb1,yb2-yb1)
		dist2=hypot(xb3-xb2,yb3-yb2)

		if dist1<=dist2:
			#puntos del lado mas corto
			xbb1,ybb1=xb1,yb1
			xbb2,ybb2=xb2,yb2
			xp1,yp1=xb2,yb2
			xp2,yp2=xb3,yb3
			#punto medio en el lado mas largo
			xcl,ycl=(xb2+xb3)/2,(yb2+yb3)/2
			dist=dist1

		else:
			#puntos del lado mas corto
			xbb1,ybb1=xb2,yb2
			xbb2,ybb2=xb3,yb3
			xp1,yp1=xb1,yb1
			xp2,yp2=xb2,yb2
			#punto medio en el lado mas largo
			xcl,ycl=(xb1+xb2)/2,(yb1+yb2)/2
			dist=dist2

		#print 'puntos ',ybb2,ybb1,xbb2,xbb1
		"""
		if xbb2-xbb1!=0:
			pendiente=float(ybb2-ybb1)/float(xbb2-xbb1)
		"""

		if xp2-xp1!=0:
			pendiente=float(yp2-yp1)/float(xp2-xp1)

		else:
			pendiente=0

		angulo_inclinacion=math.atan(pendiente)
		angulo_grados=math.degrees(angulo_inclinacion)
		print 'pendiente: ',pendiente, ' inclinacion: ',angulo_inclinacion,angulo_grados, np.radians(rect[2])
		"""
		#correccion angulo de agarre
		if (angulo_grados>-5 and angulo_grados<5) or (angulo_grados>85 and angulo_grados<95) or (angulo_grados>-85 and angulo_grados<-95):
			angulo_inclinacion=angulo_inclinacion-math.pi/2
			
		"""
		#Correccion angulo de agarre (en base al rectangulo de area minima)
		dimension=rect[1]
		angle=rect[2]
		if (dimension[0]<dimension[1] and (angle==-90 or angle==-0)):
			angulo=-np.radians(angle)
		elif (dimension[0]<dimension[1]):
			angulo=-np.radians(angle)
		else:
			angulo=-np.radians(angle)+math.pi/2
		

		#angulos_agarre.append(angulo_inclinacion)
		angulos_agarre.append(angulo)

		#Punto medio en el lado mas corto
		xcs=(xbb1+xbb2)/2
		ycs=(ybb1+ybb2)/2
		
		#Punto medio en el rectangulo de area minima
		xcentral=xcl+xcs-xb2
		ycentral=ycl+ycs-yb2

		cv2.putText(img, str(angulo_grados),(xcentral,ycentral),cv2.FONT_HERSHEY_SIMPLEX,0.4,np.array([0,200,0],dtype=np.uint8).tolist(),1,8)

		puntos_agarre.append((xcentral,ycentral))

		cv2.circle(img,(xcentral,ycentral),6,(255,0,255),-1)

		#circulo en el lado mas corto
		cv2.circle(img,(xcs,ycs),6,(0,255,0),-1)

		cv2.line(img,(xcentral,ycentral),(xcs,ycs),(0,255,0),5)		

		j=j+1
	return puntos_agarre

def correct_angle(rect_cont):
	#Correccion angulo de agarre (en base al rectangulo de area minima)
	dimension=rect_cont[1]
	angle=rect_cont[2]
	if (dimension[0]<dimension[1] and (angle==-90 or angle==-0)):
		angulo=-np.radians(angle)
	elif (dimension[0]<dimension[1]):
		angulo=-np.radians(angle)
	else:
		angulo=-np.radians(angle)+math.pi/2

	return angulo

#Movimiento al punto con angulo definido
def move(punto,angle):
	k=0;
	xdep=0.5
	ydep=0.8
	pto=punto.pop()
	(px,py)=pixel_to_baxter(pto,0.3)
	mover_baxter('base',[px,py,0.0],[math.pi,0,angle])
	mover_baxter('base',[px,py,-0.21],[math.pi,0,angle])

	gripper.close()
	rospy.sleep(0.2)

	mover_baxter('base',[px,py,0.0],[math.pi,0,angle])

	desfase=random.uniform(0,0.1)
	print 'desfase ',desfase

	mover_baxter('base',[xdep+desfase,ydep,0.0],[math.pi,0,angle])
	mover_baxter('base',[xdep+desfase,ydep,-0.18],[math.pi,0,angle])

	gripper.open()

	mover_baxter('base',[xdep+desfase,ydep,-0.18],[math.pi,0,angle])
	mover_baxter('base',[xdep+desfase,ydep,0.0],[math.pi,0,angle])


def ejecutar_mov(puntos):
	k=0
	xdep=0.5
	ydep=0.8

	while puntos:
		p=puntos.pop()
		(px,py)=pixel_to_baxter(p,0.3)
		print 'pixel to baxter: ',px,py
		mover_baxter('base',[px,py,0.0],[math.pi,0,angulos_agarre[k]])
		mover_baxter('base',[px,py,-0.2],[math.pi,0,angulos_agarre[k]])

		gripper.close()
		rospy.sleep(0.2)

		mover_baxter('base',[px,py,0.0],[math.pi,0,angulos_agarre[k]])

		desfase=random.uniform(0,0.1)
		print 'desfase ',desfase
		mover_baxter('base',[xdep+desfase,ydep,0.0],[math.pi,0,angulos_agarre[k]])
		mover_baxter('base',[xdep+desfase,ydep,-0.17],[math.pi,0,angulos_agarre[k]])

		gripper.open()

		mover_baxter('base',[xdep+desfase,ydep,0.0],[math.pi,0,angulos_agarre[k]])
		mover_baxter('base',[x,y,0.0],[math.pi,0,0])

		print 'puntos en la lista: ', puntos
		k=k+1		

def ajuste_posicion(xx,yy):
	mover_baxter('base',[xx,yy,0.0],[math.pi,0,0])

def grasp(crops):
	for cr in crops:
		#ctm=cv2.cvtColor(cr,cv2.COLOR_BGR2BGRA)
		grasping(cr)



####################################

####################################
#cargar modelo entrenado
longitud, altura = 100,100
modelo='./modelo/modelo-32b-20e-2000.h5'
pesos='./modelo/pesos-32b-20e-2000.h5'
cnn=load_model(modelo)
cnn.load_weights(pesos)

#Iniciar Nodo
rospy.init_node("reconocimiento", anonymous= True)

pub = rospy.Publisher('/robot/xdisplay', Image, latch=True, queue_size=10)

cam = baxter_interface.camera.CameraController("left_hand_camera")
cam.open()
cam.resolution = cam.MODES[0]
#cam.exposure            = -1             # range, 0-100 auto = -1
#cam.gain                = -1             # range, 0-79 auto = -1
#cam.white_balance_blue  = -1             # range 0-4095, auto = -1
#cam.white_balance_green = -1             # range 0-4095, auto = -1
#cam.white_balance_red   = -1             # range 0-4095, auto = -1
# camera parametecrs (NB. other parameters in open_camera)

	#inicializacion baxter
arm='left'
# Brazo a utilizar
limb           = arm
limb_interface = baxter_interface.Limb(limb)
if arm == "left":
	other_limb = "right"
else:
	other_limb = "left"
other_limb_interface = baxter_interface.Limb(other_limb)       
	# set speed as a ratio of maximum speed
limb_interface.set_joint_position_speed(0.5)
other_limb_interface.set_joint_position_speed(0.5)
gripper = baxter_interface.Gripper(arm)
gripper.calibrate()
# Pose inicial
x = 0.47
y = 0.3
z = 0.0 
roll = math.pi	#Rotacion x
pitch = 0.0	#Rotacion y	
yaw = 0.0		#Rotacion z

pose_i = [x, y, z, roll, pitch, yaw]
pose = [x, y, z, roll, pitch, yaw]

cam_calibracion = 0.0025            # 0.0025 pixeles por metro a 1 metro de distancia. Factor de correccion
cam_x_offset    = -0.045 #-0.01 #0.04              # Correccion de camara por los gripper,
cam_y_offset    = -0.160  #-0.115  #-0.015     
resolution      = 1
width           = 960               # 1280 640  960
height          = 600               # 800  400  600
	######
margen_img=50
tamano_deseado=100
puntos_pre=[]
nombres=[]
puntos_agarre=[]
angulos_agarre=[]
ct=[]

foto = None
def callback(msg):
	global foto 
	foto = cv_bridge.CvBridge().imgmsg_to_cv2(msg)

rospy.Subscriber('/cameras/left_hand_camera/image', Image , callback)

mover_baxter('base',[x,y,0.0],[math.pi,0,0])
rospy.sleep(1)
img_counter = 0

while not rospy.is_shutdown():
	#Capturar un frame
	while np.all(foto) == None:
		continue 

	frame = foto
	hh,ww = foto.shape[:2]

	roi=foto[300:hh ,0:ww]
	countours=img_process(frame)
	print 'contornos: ',len(countours)
	recortes=recortes_img(frame,countours)
	print 'recortes: ',len(recortes)
	print recortes[0].shape
	crop=recortes[0]
	crop2=crop[:,:,:3]
	#cv2.imwrite('crop.jpg',crop2)
	grasping(crop2.astype(np.int32))
	rect_grasp=get_points()

	#############Calculo del centro del rectangulo de grasping
	puntx1,punty1=rect_grasp[0]
	puntx2,punty2=rect_grasp[1]
	puntx3,punty3=rect_grasp[2]
	dist1=hypot(puntx2-puntx1,punty2-punty1)
	dist2=hypot(puntx3-puntx2,punty3-punty2)
	xd,yd=((puntx1+puntx2)/2,(punty1+punty2)/2)
	xd2,yd2=((puntx2+puntx3)/2,(punty2+punty3)/2)
	xd,xd2,yd,yd2=int(xd),int(xd2),int(yd),int(yd2)
	pxmedio,pymedio=int(xd2+xd-puntx2),int(yd2+yd-punty2)
	puntx1,puntx2,puntx3=int(puntx1),int(puntx2),int(puntx3)
	punty1,punty2,punty3=int(punty1),int(punty2),int(punty3)
	cv2.circle(crop,(pxmedio,pymedio),6,(0,180,0),-1)
	cv2.circle(crop,(puntx1,punty1),6,(255,0,255),-1)
	cv2.circle(crop,(puntx2,punty2),6,(255,0,255),-1)
	cv2.circle(crop,(puntx3,punty3),6,(255,0,255),-1)
	cv2.circle(crop,(xd,yd),6,(0,0,255),-1)
	cv2.circle(crop,(xd2,yd2),6,(0,0,255),-1)

	cv2.imwrite('crop.jpg',crop)
	#############

	print 'puntos grasp ',rect_grasp
	print 'punto: ',rect_grasp[0], rect_grasp[1]
	(puntox,puntoy)=pixel_to_baxter(rect_grasp[0],0.3)
	print 'valor x e y ',xi,yi
	#mover_baxter('base',[puntox,puntoy,0],[math.pi,0,0])
	print 'suma de punto= ', rect_grasp[0]+rect_grasp[1]
	nombres=prediccion_obj(recortes)
	print 'nombres: ',len(nombres)
	puntos=info_and_angles(frame,countours,nombres)
	print 'puntos: ',len(puntos)
	print puntos

	print 'puntos pre: ', puntos_pre

	copia_puntos=puntos_pre
	rospy.sleep(2)

	imagen_camara=cv2.resize(frame,(1024,600))
	#print frame.shape[:2]
	cv2.imwrite('cam.jpg',imagen_camara)
	send_image('cam.jpg')

	#Prueba, acercamiento por objeto
	while copia_puntos:   #se acerca a cada punto donde detecto un objeto el analisis inicial
		point=copia_puntos.pop()
		(pointx,pointy)=pixel_to_baxter(point,0.3)
		#mover_baxter('base',[pointx-0.05,pointy+0.05,0.0],[math.pi,0,0])
		contorno=ucontorno(foto)  #recalcula el contorno y lo retorna
		print 'contorno: ',len(contorno)
		rospy.sleep(1)
		cv2.imwrite('ct.jpg',foto)

		#Calcula el Rectangulo de area minima del contorno calculado
		if len(contorno)!=0:
			rect=cv2.minAreaRect(contorno[0])			
		else: 
			continue
		#Guarda el centro del rectangulo minimo
		ppp=[]
		ppp.append(rect[0])
		#Actualiza la nueva pose donde se encuentra el brazo
		pose_i = [pointx-0.05, pointy+0.05, z, roll, pitch, yaw]
		pose = [pointx-0.05, pointy+0.05, z, roll, pitch, yaw]
		angulo_corregido=correct_angle(rect)  #Corrige el angulo en la nueva posicion

		print ppp
		print angulo_corregido
		#move(ppp,angulo_corregido)

		#Actualizo posicion inicial
		pose_i = [x, y, z, roll, pitch, yaw]
		pose = [x, y, z, roll, pitch, yaw]
		#mover_baxter('base',[x,y,0.0],[math.pi,0,0])
		print 'centro: ',rect[0]
	
	print 'frame shape',frame.shape
	cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
	print 'nuevo tamano',frame.shape
	frame2=foto[:,:,:3]
	cv2.imwrite('ca.jpg',frame2)
	print 'nuevo tamano 2',frame2.shape
	print 'type ',type(frame2) 
	#grasping(frame2.astype(np.int32))
	cv2.imwrite('test.jpg',frame2)

	cam2= cv2.imread("grasp.jpg")
	imagen_camara2=cv2.resize(cam2,(1024,600))
	cv2.imwrite('cam2.jpg',imagen_camara2)
	#send_image('cam2.jpg')

	del ct[:]  #ct= arreglo que almacena contorno detectado temporalmente


	rospy.sleep(0.5)

	#ejecutar_mov(puntos)

	#ejecutar_mov(puntos)

	#mover_baxter('base',[x,y,0.0],[math.pi,0,0])

	#plt.imshow(roi)
	#plt.show()

	#para limpiar la lista de puntos
	del puntos_agarre[:]
	del angulos_agarre[:]
	del puntos_pre[:]

	k=cv2.waitKey(1)

	if k%256==27:
		print("Escape hit, closing...")
		break
	elif k%256==32:
		img_name = "opencv_frame_{}.png".format(img_counter)
		cv2.imwrite(img_name, frame)
		print("{} written!".format(img_name))
		img_counter += 1

	rospy.sleep(2)
