#!/usr/bin/env python
import numpy as np
import math 
import cv2
import rospy
import baxter_interface
import cv_bridge
import roslib
import tf
from sensor_msgs.msg import Image
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from math import hypot

#cargar modelo entrenado
longitud, altura = 100,100
modelo='./modelo/modelo-32b-20e-2000.h5'
pesos='./modelo/pesos-32b-20e-2000.h5'
cnn=load_model(modelo)
cnn.load_weights(pesos)

tamano_deseado=100
#####

def prediccion(file):
	#x = load_img(file,target_size=(longitud,altura))
	#x = img_to_array(x)
	#x = np.expand_dims(x,axis=0)
	####Esta transformacion se hace para eliminar un canal a la imagen (imagen = (100,100,4))
	#z=cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
	#y=cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)
	####
	y=cv2.cvtColor(file,cv2.COLOR_BGRA2BGR)
	x=cv2.resize(y,(longitud,altura))
	#cv2.imwrite('crop.jpg',x)
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


rospy.init_node("reconocimiento", anonymous= True)

#####################
#####################
#Funcion Movimiento Baxter
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
x = 0.6
y = 0.3
z = 0.0 
roll = math.pi	#Rotacion x
pitch = 0.0	#Rotacion y	
yaw = 0.0		#Rotacion z

pose_i = [x, y, z, roll, pitch, yaw]
pose = [x, y, z, roll, pitch, yaw]

cam_calibracion = 0.0025            # 0.0025 pixeles por metro a 1 metro de distancia. Factor de correccion
cam_x_offset    = 0.04              # Correccion de camara por los gripper,
cam_y_offset    = -0.015       
resolution      = 1
width           = 960               # 1280 640  960
height          = 600               # 800  400  600


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

#####################
#####################

def pixel_to_baxter(px, dist):
	x = ((px[1] - (height / 2)) * cam_calibracion * dist)+ pose_i[0] + cam_x_offset
	y = ((px[0] - (width / 2)) * cam_calibracion * dist)+ pose_i[1] + cam_y_offset
	
	return (x, y)

def QtoE(): #Quaternion to Euler. Converts Quaternion angles(x, y, z, w) into Euler angles (x, y ,z) and prints them
	euler = tf.transformations.euler_from_quaternion(limb_interface.endpoint_pose()['orientation'])
	print ("Arm positions and Quaternion angles")
	print (limb_interface.endpoint_pose())
	print ("Arm Euler angles: ", euler)


cam = baxter_interface.camera.CameraController("left_hand_camera")

cam.open()

cam.resolution = cam.MODES[0]

foto = None

def callback(msg):
	global foto 
	foto = cv_bridge.CvBridge().imgmsg_to_cv2(msg)

rospy.Subscriber('/cameras/left_hand_camera/image', Image , callback)

puntos_agarre=[]
angulos_agarre=[]

#############
mover_baxter('base',[x,y,0.0],[math.pi,0,0])
#QtoE()
#############

while not rospy.is_shutdown():
	while np.all(foto)== None:
		continue
	#rospy.sleep(5)
	frame=foto
	hh,ww = foto.shape[:2]
	#print (x,y)
	roi=foto[300:hh ,0:ww]
	#cv2.imshow("ROI",roi)
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	dimensiones = gray.shape
	blur = cv2.GaussianBlur(gray,(5,5),0)
	canny = cv2.Canny(blur,50,200)
	#cv2.imshow("canny",canny)

	#Morphologic, para completar bordes
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	dilated = cv2.dilate(canny,kernel)

	#cv2.imshow('camara dilatada',dilated)

	(contornos,_) = cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	print len(contornos)

	for c in contornos:
		#cv2.drawContours(roi,[c],-1,(0,0,255),2)
		#dibujar rectangulos sobre ls bordes
		x,y,w,h = cv2.boundingRect(c)
		#cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)

		#rospy.sleep(0.03)
		rospy.sleep(0.3)
	
	#rospy.sleep(5)
	margen_img=10
	#cv2.imshow('cam',frame)
	for c in contornos:
		area=cv2.contourArea(c)
		x,y,w,h=cv2.boundingRect(c)
		xc,yc=x+w/2,y+h/2
		#cv2.circle(roi,(xc,yc),6,(200,255,0),-1)
		puntos_agarre.append((xc,yc))
		area=w*h
		print area
		if area<=1000:
			continue

		yi=y-margen_img
		yf=y+h+margen_img
		xi=x-margen_img
		xf=x+w+margen_img

		if yi<0:
			yi=0
		if xi<0:
			xi=0

		crop=roi[yi:yf, xi:xf]

		#crop=foto[y:y+h,x:x+w]
		#crop=roi[y:y+h,x:x+w]
		#cv2.imshow("crop",crop)
		rospy.sleep(1)
		cv2.imshow('crop',crop)

		##############
		##Codigo que permite hacer resize manteniendo el ratio
		##Hace lo mismo que la funcion get_square() pero en RGB

		desired_size=tamano_deseado

		old_size=crop.shape[:2]

		ratio=float(desired_size)/max(old_size)
		new_size = tuple([int(o*ratio) for o in old_size])
		im = cv2.resize(crop, (new_size[1], new_size[0]))

		delta_w = desired_size - new_size[1]
		delta_h = desired_size - new_size[0]
		top, bottom = delta_h//2, delta_h-(delta_h//2)
		left, right = delta_w//2, delta_w-(delta_w//2)

		color = [255, 255, 255]
		
		new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
		#new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_REPLICATE,value=color)
		#imagenes.append(new_im)

		#cv2.imshow("imagen", new_im)
		#cv2.imwrite('new.jpg',new_im)

		#############

		nombre=prediccion(new_im)
		print 'nombre: ',nombre

		cv2.rectangle(roi, (x,y),(x+w,y+h), (255,0,0),2)
		cv2.putText(roi, nombre,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,np.array([0,0,0],dtype=np.uint8).tolist(),1,8)

		#Rectangulo de area minima
		rect=cv2.minAreaRect(c)
		box=cv2.cv.BoxPoints(rect)
		box=np.int0(box)

		cv2.drawContours(roi,[box],0,(0,0,255),2)

		##Puntos rectangulo minimo, se toma como primer punto el que este mas abajo en el eje y
		##luego los siguientes se toman en el sentido de las agujas del reloj
		xb1,yb1=box[0][0],box[0][1]
		xb2,yb2=box[1][0],box[1][1]
		xb3,yb3=box[2][0],box[2][1]

		dist1=hypot(xb2-xb1,yb2-yb1)
		dist2=hypot(xb3-xb2,yb3-yb2)

		if dist1<=dist2:
			xbb1,ybb1=xb1,yb1
			xbb2,ybb2=xb2,yb2
			#punto medio en el lado mas largo
			xcl,ycl=(xb2+xb3)/2,(yb2+yb3)/2
			dist=dist1

		else:
			xbb1,ybb1=xb2,yb2
			xbb2,ybb2=xb3,yb3
			#punto medio en el lado mas largo
			xcl,ycl=(xb1+xb2)/2,(yb1+yb2)/2
			dist=dist2

		#print 'puntos ',ybb2,ybb1,xbb2,xbb1
		if xbb2-xbb1!=0:
			pendiente=float(ybb2-ybb1)/float(xbb2-xbb1)

		else:
			pendiente=1

		angulo_inclinacion=math.atan(pendiente)
		angulo_grados=math.degrees(angulo_inclinacion)
		print 'pendiente: ',pendiente, ' inclinacion: ',angulo_inclinacion,angulo_grados
		angulos_agarre.append(angulo_inclinacion)

		#Punto medio en el lado mas corto
		xcs=(xbb1+xbb2)/2
		ycs=(ybb1+ybb2)/2
		
		#Punto medio en el rectangulo de area minima
		xcentral=xcl+xcs-xb2
		ycentral=ycl+ycs-yb2

		cv2.circle(roi,(xcentral,ycentral),6,(255,0,255),-1)

		#circulo en el lado mas corto
		cv2.circle(roi,(xcs,ycs),6,(0,255,0),-1)

		cv2.line(roi,(xcentral,ycentral),(xcs,ycs),(0,255,0),5)

		#mover_baxter('base',[x,y,z],[math.pi,0,angulo_inclinacion])

	print len(puntos_agarre)


	k=0
#dist=22cm
	for i in puntos_agarre:
		print 'punto de agarre: ', i
		(px,py)=pixel_to_baxter(i,0.25)
		print 'pixel to baxter: ', px,py
		#mover_baxter('base',[px,py,0.0],[math.pi,0,angulos_agarre[k]])
		print 'movimiento, angulo de giro ', angulo_inclinacion
		#mover_baxter('base',[px,py,-0.2],[math.pi,0,angulos_agarre[k]])

		k=k+1

	cv2.imshow('cam',frame)
	#cv2.imshow('roi',roi)
	#print len(contornos)

	#para limpiar la lista de puntos
	del puntos_agarre[:]
	del angulos_agarre[:]

	k=cv2.waitKey(5) & 0xFF
	if k==27:
		break

cv2.destroyAllWindows()