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
from grasp_image import get_points,init_model,prediccion_grasp
from collections import Counter
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
		herramienta='Tijera'
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
	#canny= cv2.Canny(blur,50,200)
	canny= cv2.Canny(blur,20,100)

	#Morphologic, para completar bordes
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	dilated = cv2.dilate(canny,kernel)

	cv2.imwrite('cam_dilated.jpg',dilated)
	
	(contornos,_) = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
	canny= cv2.Canny(blur,20,100)

	#Morphologic, para completar bordes
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	dilated = cv2.dilate(canny,kernel)

	centro_garra=(640,400)
	distancias=[]
	centros=[]
	ctn=[]
	(contornos,_) = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for c in contornos:
		x,y,w,h=cv2.boundingRect(c)
		#Rectangulo de area minima
		rect=cv2.minAreaRect(c)
		box=cv2.cv.BoxPoints(rect)
		box=np.int0(box)

		area=w*h
		#Se limita la zona para ignorar contornos en los bordes
		#if area<=1000:
		if area<=1000 or y<300 or x<400 or x>880:
			continue

		#cv2.rectangle(image, (x,y),(x+w,y+h),(255,0,255),2)
		centros.append(box[0])
		ct.append(c)

	if len(centros)>1:
		i=0
		for c in centros:
			#dist= hypot(centro_garra[0]-centros[i][0],centro_garra[1]-centros[i][1])
			dist= hypot(centro_garra[0]-c[0],centro_garra[1]-c[1])
			distancias.append(dist)
			i=i+1
		#print 'distancias a cada punto ',distancias
		dist_min=min(distancias)
		cnt_final=ct[distancias.index(dist_min)]
		ctn.append(cnt_final)
	else:
		ctn=ct

	return ctn

xi=0
yi=0

xcrop=0
ycrop=0
ptos_corte=[]
pto_corte_preciso=[]

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
		######Prueba 
		xcrop,ycrop=xi,yi
		ptos_corte.append((xcrop,ycrop))
		######
		if yi<0:
			yi=0
		if xi<0:
			xi=0

		crop=image[yi:yf,xi:xf]
		recortes.append(crop)
		cv2.imwrite('crop1.jpg',crop)

	return recortes

def urecorte(image,contorno):
	rct=[]
	for c in contorno:
		x,y,w,h=cv2.boundingRect(c)
		xc,yc=x+w/2,y+h/2
		global xi,yi
		yi=y-margen_img+10
		yf=y+h+margen_img-10
		xi=x-margen_img+10
		xf=x+w+margen_img-10
		###
		xcrop,ycrop=xi,yi
		pto_corte_preciso.append((xcrop,ycrop))
		###
		if yi<0:
			yi=0
		if xi<0:
			xi=0
		crop=image[yi:yf,xi:xf]
		rct.append(crop)
		cv2.imwrite('cropu.jpg',crop)
	return rct

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
		
	new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_REPLICATE,value=color)
	cv2.imwrite('reescalado.jpg',new_im)
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
		cv2.putText(img, nombres[j],(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,np.array([0,0,0],dtype=np.uint8).tolist(),1,8)

		#Rectangulo de area minima
		rect=cv2.minAreaRect(c)
		box=cv2.cv.BoxPoints(rect)
		box=np.int0(box)
		#cv2.drawContours(img,[box],0,(0,0,255),2)

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

		puntos_agarre.append((xcentral,ycentral))

		#cv2.circle(img,(xcentral,ycentral),6,(255,0,255),-1)

		#circulo en el lado mas corto
		#cv2.circle(img,(xcs,ycs),6,(0,255,0),-1)

		#cv2.line(img,(xcentral,ycentral),(xcs,ycs),(0,255,0),5)		

		j=j+1
	return puntos_agarre

def correct_angle(rect_cont):
	#Correccion angulo de agarre (en base al rectangulo de area minima)
	dimension=rect_cont[1]
	print 'dimension: ',dimension
	angle=rect_cont[2]
	if (dimension[0]<dimension[1] and (angle==-90 or angle==-0)):
		angulo=-np.radians(angle)
	elif (dimension[0]<dimension[1]):
		angulo=-np.radians(angle)
	else:
		angulo=-np.radians(angle)+math.pi/2
	return angulo

####*****Solucion Parche*******###### Por ahora no es necesaria, ya esta arreglado (5/12)
def cvt_box(box): 
	blank_image=np.zeros((800,1280,3),np.uint8)
	cv2.line(blank_image,(int(box[0][0]),int(box[0][1])), (int(box[1][0]),int(box[1][1])), color=(0,255,0), thickness=5)
	cv2.line(blank_image,(int(box[1][0]),int(box[1][1])), (int(box[2][0]),int(box[2][1])), color=(0,255,0), thickness=5)
	cv2.line(blank_image,(int(box[2][0]),int(box[2][1])), (int(box[3][0]),int(box[3][1])), color=(0,255,0), thickness=5)
	cv2.line(blank_image,(int(box[3][0]),int(box[3][1])), (int(box[0][0]),int(box[0][1])), color=(0,255,0), thickness=5)
	
	gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
	gauss = cv2.GaussianBlur(gray, (5,5), 0)
	canny = cv2.Canny(gauss, 50, 200)
	#(_, contornos, _) = cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	(contornos, _) = cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	cnt=contornos[0]

	rect = cv2.minAreaRect(cnt)
	box=cv2.cv.BoxPoints(rect)
	box = np.int0(box)
	cv2.drawContours(blank_image,[box],0,(0,0,255),2)
	print 'angulo de la caja rect: ',rect[2]
	cv2.imwrite('blank_i.jpg',blank_image)

	return rect

#Movimiento al punto con angulo definido
def move1(punto,angle):
	k=0;
	xdep=0.5
	ydep=0.8
	#pto=punto.pop()
	#(px,py)=pixel_to_baxter(pto,0.3)
	(px,py)=pixel_to_baxter(punto,0.3)
	mover_baxter('base',[px,py,0.0],[math.pi,0,angle])
	mover_baxter('base',[px,py,-0.22],[math.pi,0,angle])

	gripper.close()

	rospy.sleep(0.4)

	mover_baxter('base',[px,py,0.0],[math.pi,0,angle])

	desfase=random.uniform(0,0.1)
	print 'desfase ',desfase

	mover_baxter('base',[xdep+desfase,ydep,0.0],[math.pi,0,angle])
	mover_baxter('base',[xdep+desfase,ydep,-0.19],[math.pi,0,angle])

	gripper.open()

	mover_baxter('base',[xdep+desfase,ydep,0.0],[math.pi,0,angle])
	mover_baxter('base',[x,y,0.0],[math.pi,0,0])

def move(punto,angle,nombre):
	k=0;
	xdep=0.5
	ydep=0.8
	#pto=punto.pop()
	#(px,py)=pixel_to_baxter(pto,0.3)

	if (nombre=='Alicate' or nombre=='Cuchillo' or nombre=='Destornillador' or nombre=='Llave'):
		deposito=(0.5470207284933092, 0.7550064358491599)
	
	if (nombre=='Calculadora' or nombre=='Gamepad' or nombre=='Mouse'):
		deposito=(0.31195328392783256, 0.6652463368344972)

	if (nombre=='Martillo' or nombre=='Taladro' or nombre=='Tijera' or nombre=='Reloj'):
		deposito=(0.309293240857769, 0.8451825272761175)



	(px,py)=pixel_to_baxter(punto,0.3)
	mover_baxter('base',[px,py,0.0],[math.pi,0,angle])
	mover_baxter('base',[px,py,-0.215],[math.pi,0,angle])
	rospy.sleep(0.2)

	gripper.close()

	rospy.sleep(0.6)
	mover_baxter('base',[px,py,0.0],[math.pi,0,angle])

	if gripper.force()==0:
		print 'no hay nada en el gripper'
		gripper.open()	
		mover_baxter('base',[x,y,0.0],[math.pi,0,0])

	else:	
		mover_baxter('base',[deposito[0],deposito[1],0.0],[math.pi,0,angle])
		mover_baxter('base',[deposito[0],deposito[1],-0.19],[math.pi,0,angle])

		gripper.open()

		mover_baxter('base',[deposito[0],deposito[1],0.0],[math.pi,0,angle])
		mover_baxter('base',[x,y,0.0],[math.pi,0,0])


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

def centro_grasp(rect_grasp,xycroppreciso,xycrop):
	####Calculo del centro del rectangulo de grasping
	xcrop,ycrop=xycroppreciso
	xxcrop,yycrop=xycrop
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

	#coordenadas para dibujar sobre el frame desplazado
	point_centerx,point_centery=pxmedio+xcrop,pymedio+ycrop
	rect_grasp[0][0],rect_grasp[1][0],rect_grasp[2][0],rect_grasp[3][0]=rect_grasp[0][0]+xcrop,rect_grasp[1][0]+xcrop,rect_grasp[2][0]+xcrop,rect_grasp[3][0]+xcrop
	rect_grasp[0][1],rect_grasp[1][1],rect_grasp[2][1],rect_grasp[3][1]=rect_grasp[0][1]+ycrop,rect_grasp[1][1]+ycrop,rect_grasp[2][1]+ycrop,rect_grasp[3][1]+ycrop

	cv2.circle(foto,(point_centerx,point_centery),6,(0,250,0),-1)
	cv2.line(foto, tuple(rect_grasp[0].astype(int)), tuple(rect_grasp[1].astype(int)), color=(0,255,0), thickness=5)
	cv2.line(foto, tuple(rect_grasp[1].astype(int)), tuple(rect_grasp[2].astype(int)), color=(0,0,255), thickness=5)
	cv2.line(foto, tuple(rect_grasp[2].astype(int)), tuple(rect_grasp[3].astype(int)), color=(0,255,0), thickness=5)
	cv2.line(foto, tuple(rect_grasp[3].astype(int)), tuple(rect_grasp[0].astype(int)), color=(0,0,255), thickness=5)

	
	#coordenadas para dibujar de manera correcta sobre el frame general
	p_centrox_dibujo,p_centroy_dibujo=pxmedio+xxcrop,pymedio+yycrop
	rect_grasp[0][0],rect_grasp[1][0],rect_grasp[2][0],rect_grasp[3][0]=rect_grasp[0][0]+xxcrop-xcrop,rect_grasp[1][0]+xxcrop-xcrop,rect_grasp[2][0]+xxcrop-xcrop,rect_grasp[3][0]+xxcrop-xcrop
	rect_grasp[0][1],rect_grasp[1][1],rect_grasp[2][1],rect_grasp[3][1]=rect_grasp[0][1]+yycrop-ycrop,rect_grasp[1][1]+yycrop-ycrop,rect_grasp[2][1]+yycrop-ycrop,rect_grasp[3][1]+yycrop-ycrop
	
	#Dibujo
	cv2.circle(frame3,(p_centrox_dibujo,p_centroy_dibujo),6,(0,250,0),-1)
	cv2.line(frame3, tuple(rect_grasp[0].astype(int)), tuple(rect_grasp[1].astype(int)), color=(0,255,0), thickness=5)
	cv2.line(frame3, tuple(rect_grasp[1].astype(int)), tuple(rect_grasp[2].astype(int)), color=(0,0,255), thickness=5)
	cv2.line(frame3, tuple(rect_grasp[2].astype(int)), tuple(rect_grasp[3].astype(int)), color=(0,255,0), thickness=5)
	cv2.line(frame3, tuple(rect_grasp[3].astype(int)), tuple(rect_grasp[0].astype(int)), color=(0,0,255), thickness=5)


	return point_centerx,point_centery

##################################################

#cargar modelo entrenado
longitud, altura = 100,100
modelo='./modelo/modelo-32b-25e-2000-corregido-rms.h5'
pesos='./modelo/pesos-32b-25e-2000-corregido-rms.h5'
cnn=load_model(modelo)
cnn.load_weights(pesos)

#Iniciar Nodo
rospy.init_node("reconocimiento", anonymous= True)

pub = rospy.Publisher('/robot/xdisplay', Image, latch=True, queue_size=10)

cam = baxter_interface.camera.CameraController("left_hand_camera")
cam.open()
cam.resolution = cam.MODES[0]
cam.exposure            = -1             # range, 0-100 auto = -1
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

model_grasp=init_model()


while not rospy.is_shutdown():
	#Capturar un frame
	while np.all(foto) == None:
		continue 

#	deposito1=[x=0.5470207284933092,y=0.7550064358491599]
#	deposito2=[x=0.31195328392783256, y=0.6652463368344972]
#	deposito3=[x=0.309293240857769, y=0.8451825272761175]

	frame = foto
	hh,ww = foto.shape[:2]
	roi=foto[300:hh ,0:ww]
	countours=img_process(frame)
	print 'contornos: ',len(countours)
	recortes=recortes_img(frame,countours)
	print 'recortes: ',len(recortes)
	nombres=prediccion_obj(recortes)
	print 'nombres: ',len(nombres)
	fps=foto
	puntos=info_and_angles(fps,countours,nombres)
	cv2.imwrite('fps.jpg',fps)
	print 'puntos: ',len(puntos)
	print puntos
	
	frame3=foto

	objetos_solicitados=[]
	print 'Se han encontrado ',len(countours),' objetos'
	print 'Objetos encontrados: '
	sss=Counter(nombres)
	print sss
	print 'tipo de dato ',type(sss)



	print 'Modos disponible:' 
	print '1-Busqueda por seleccion, 2-Busqueda automatica, 3-Busqueda Basica'
	modo=input("Seleccione modo de busqueda ")
	while(modo!=1 and modo!=2 and modo!=3):
		modo=raw_input("Seleccione modo de busqueda ")
		if modo>=4:
			break

	if modo==1:
	###############Modo de busqueda por seleccion, se elige el objeto a recoger y se ejecuta la accion
		inp=raw_input("Que desea recoger? ")
		j=0
		for i in nombres:
			if i==inp:
				objetos_solicitados.append(j)
			j=j+1
		print objetos_solicitados

		while objetos_solicitados:
			obs=objetos_solicitados.pop()
			punto_objetivo=ptos_corte[obs]
			recorte_sol=recortes[obs]
			print 'recorte tamano ',recorte_sol.shape
			#punto_objetivo=puntos[obs]
			(pxo,pyo)=pixel_to_baxter(punto_objetivo,0.3)
			mover_baxter('base',[pxo+0.05,pyo+0.1,0.0],[math.pi,0,0])
			rospy.sleep(1)
			cv2.circle(foto,(640,400),6,(255,123,255),-1)
			cv2.imwrite('acercamiento.jpg',foto)
			ctrn=ucontorno(foto)
			print 'size ucontorno ',len(ctrn)
			recorte_aj=urecorte(foto,ctrn)
			print 'size recorte_aj ',len(recorte_aj)
			rct_u=recorte_aj.pop()
			rctu=rct_u
			rct_u=rct_u[:,:,:3]
			punto_preciso_corte=pto_corte_preciso.pop()
			rospy.sleep(0.5)
			prediccion_grasp(rct_u.astype(np.int32),model_grasp)
			box=get_points()
			print 'caja de puntos grasp ',box
			cv2.imwrite('rct_u.jpg',rctu)
			print rctu.shape
			size_recorte_p=rctu.shape
			size_recorte_i=recorte_sol.shape
			##########Calcular diferencia de pixeles entre los dos recortes
			dif=(size_recorte_i[1]-size_recorte_p[1],size_recorte_i[0]-size_recorte_p[0])
			print dif
			##########
			centro_agarre=centro_grasp(box,punto_preciso_corte,punto_objetivo)
			#centro_agarre=centro_grasp(box,punto_preciso_corte,punto_objetivo)
			(pointx,pointy)=pixel_to_baxter(centro_agarre,0.3)
			box[4]=box[4]*-1
			pose_i = [pxo+0.05, pyo+0.1, z, roll, pitch, yaw]
			pose = [pxo+0.05, pyo+0.1, z, roll, pitch, yaw]
			cv2.imwrite('acercamiento.jpg',foto)
			move(centro_agarre,box[4],inp)
			cv2.imwrite('frame3.jpg',frame3)
			########
			imagen_camara=cv2.resize(frame3,(1024,600))
			cv2.imwrite('cam.jpg',imagen_camara)
			send_image('cam.jpg')
			########
			#Actualizo posicion inicial
			pose_i = [x, y, z, roll, pitch, yaw]
			pose = [x, y, z, roll, pitch, yaw]

	if modo==2:
	################Modo automatico, se desplaza a cada recorte y recoge el objeto
		while recortes:
			recort=recortes.pop()
			rec=recort[:,:,:3]
			n_obj=nombres.pop()
			#cv2.imwrite('rec.jpg',rec)
			punto_corte=ptos_corte.pop()
			#############Prueba acercamiento por recorte para mejorar precision del brazo
			(pxct,pyct)=pixel_to_baxter(punto_corte,0.3)
			mover_baxter('base',[pxct+0.05,pyct+0.1,0.0],[math.pi,0,0])
			cv2.imwrite('acercamiento.jpg',foto)
			ctn=ucontorno(foto)
			rec_ajustado=urecorte(foto,ctn)
			rcort=rec_ajustado.pop()
			rcort=rcort[:,:,:3]
			punto_preciso_corte=pto_corte_preciso.pop()
			rospy.sleep(1)
			prediccion_grasp(rcort.astype(np.int32),model_grasp)
			box=get_points()
			centro_agarre=centro_grasp(box,punto_preciso_corte,punto_corte)
			(pointx,pointy)=pixel_to_baxter(centro_agarre,0.3)
			box[4]=box[4]*-1 #Correccion de angulo, ya que la funcion entrega angulo con signo cambiado
			pose_i = [pxct+0.05, pyct+0.1, z, roll, pitch, yaw]
			pose = [pxct+0.05, pyct+0.1, z, roll, pitch, yaw]
			cv2.imwrite('frame3.jpg',frame3)
			move(centro_agarre,box[4],n_obj)
			#Actualizo posicion inicial
			pose_i = [x, y, z, roll, pitch, yaw]
			pose = [x, y, z, roll, pitch, yaw]

			#############
			#prediccion_grasp(rec.astype(np.int32),model_grasp)
			#box=get_points()
			#print 'caja puntos prueba error ',box
			#centro_agarre=centro_grasp(box,punto_corte)
			#(pointx,pointy)=pixel_to_baxter(centro_agarre,0.3)
			#box[4]=box[4]*-1 #Correccion de angulo, ya que la funcion entrega angulo con signo cambiado
			#print "angulo grasp ",box[4]
			#move(centro_agarre,box[4])

	if modo==3:
		copia_puntos=puntos_pre
		print len(puntos_pre)
		#Prueba, acercamiento por objeto
		while copia_puntos:   #se acerca a cada punto donde detecto un objeto el analisis inicial
			point=copia_puntos.pop()
			name_obj=nombres.pop()
			(pointx,pointy)=pixel_to_baxter(point,0.3)
			mover_baxter('base',[pointx-0.05,pointy+0.05,0.0],[math.pi,0,0])
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
			move(rect[0],angulo_corregido,name_obj)

			#Actualizo posicion inicial
			pose_i = [x, y, z, roll, pitch, yaw]
			pose = [x, y, z, roll, pitch, yaw]
			print 'centro: ',rect[0]


	cv2.imwrite('frame3.jpg',frame3)

	rospy.sleep(2)

	imagen_camara=cv2.resize(frame3,(1024,600))
	#print frame.shape[:2]
	cv2.imwrite('cam.jpg',imagen_camara)
	send_image('cam.jpg')

	print 'frame shape',frame.shape
	cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
	print 'nuevo tamano',frame.shape
	frame2=foto[:,:,:3]
	cv2.imwrite('ca.jpg',frame2)
	print 'nuevo tamano 2',frame2.shape
	print 'type ',type(frame2) 
	#grasping(frame2.astype(np.int32))
	#cv2.imwrite('test.jpg',frame2)


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
	del ptos_corte[:]

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
