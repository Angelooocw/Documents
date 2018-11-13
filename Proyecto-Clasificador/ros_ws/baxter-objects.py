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

def QtoE(): #Quaternion to Euler. Converts Quaternion angles(x, y, z, w) into Euler angles (x, y ,z) and prints them
	euler = tf.transformations.euler_from_quaternion(limb_interface.endpoint_pose()['orientation'])
	print ("Arm positions and Quaternion angles")
	print (limb_interface.endpoint_pose())
	print ("Arm Euler angles: ", euler)

#Detecta Bordes de objetos de la camara y retorna los contornos validos
def img_process(image):
	hh,ww=image.shape[:2]
	roi=image[300:hh, 0:ww]

	gray= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	blur= cv2.GaussianBlur(gray,(5,5),0)
	canny= cv2.Canny(blur,50,200)

	#Morphologic, para completar bordes
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	dilated = cv2.dilate(canny,kernel)

	(contornos,_) = cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cont_validos=[]
	for c in contornos:
		x,y,w,h=cv2.boundingRect(c)
		area=w*h
		if area<=1000:
			continue
		cont_validos.append(c)

	return cont_validos

#Realiza recortes por cada contorno detectado
def recortes_img(image, contornos):
	recortes=[]
	for c in contornos:
		x,y,w,h=cv2.boundingRect(c)
		xc,yc=x+w/2,y+h/2
		puntos_agarre.append((xc,yc))
		"""area=w*h
								if area<=1000:
									continue"""

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


#Dibuja sobre la imagen con respecto a los objetos y calcula angulo de inclinacion
def info_and_angles(img,contornos,nombres):
	j=0
	for c in contornos:
		x,y,w,h=cv2.boundingRect(c)
		cv2.rectangle(img, (x,y+300),(x+w,y+h+300),(255,0,0),2)
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

		cv2.circle(img,(xcentral,ycentral),6,(255,0,255),-1)

		#circulo en el lado mas corto
		cv2.circle(img,(xcs,ycs),6,(0,255,0),-1)

		cv2.line(img,(xcentral,ycentral),(xcs,ycs),(0,255,0),5)		

		j=j+1

def ejecutar_mov(puntos_agarre):
	k=0
	for p in puntos_agarre:
		print 'punto de agarre: ', p
		(px,py)=pixel_to_baxter(p,0.25)
		print 'pixel to baxter: ', px,py
		mover_baxter('base',[px,py,0.0],[math.pi,0,angulos_agarre[k]])
		print 'movimiento, angulo de giro ', angulo_inclinacion
		mover_baxter('base',[px,py,-0.2],[math.pi,0,angulos_agarre[k]])
		k=k+1

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
cam = baxter_interface.camera.CameraController("left_hand_camera")
cam.open()
cam.resolution = cam.MODES[0]
#cam.exposure            = -1             # range, 0-100 auto = -1
#cam.gain                = -1             # range, 0-79 auto = -1
#cam.white_balance_blue  = -1             # range 0-4095, auto = -1
#cam.white_balance_green = -1             # range 0-4095, auto = -1
#cam.white_balance_red   = -1             # range 0-4095, auto = -1
# camera parametecrs (NB. other parameters in open_camera)
cam_calib    = 0.0025                     # meters per pixel at 1 meter
cam_x_offset = 0.0                       # camera gripper offset
cam_y_offset = 0.0
width        = 960 #640 960                       # Camera resolution
height       = 600 #400 600
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
	######
margen_img=10
tamano_deseado=100
nombres=[]
puntos_agarre=[]
angulos_agarre=[]
print 'pase la inicializacion de variables'
foto = None
def callback(msg):
	global foto 
	foto = cv_bridge.CvBridge().imgmsg_to_cv2(msg)

rospy.Subscriber('/cameras/left_hand_camera/image', Image , callback)

img_counter = 0
	
print 'suscripcion'
while not rospy.is_shutdown():
	#Capturar un frame
	while np.all(foto) == None:
		#print "hola"
		continue 

	frame = foto
	countours=img_process(frame)
	print len(countours)
	recortes=recortes_img(frame,countours)
	print len(recortes)
	nombres=prediccion_obj(recortes)
	print len(nombres)
	info_and_angles(frame,countours,nombres)


	#Mostrar la imagen
	cv2.imshow('Imagen', frame)

	k=cv2.waitKey(1)

	if k%256==27:
		print("Escape hit, closing...")
		break
	elif k%256==32:
		img_name = "opencv_frame_{}.png".format(img_counter)
		cv2.imwrite(img_name, frame)
		print("{} written!".format(img_name))
		img_counter += 1

cv2.destroyAllWindows()
