#!/usr/bin/env python
# -*- coding: cp1252 -*-
import rospy
import baxter_interface
import roslib
import cv2
import numpy as np
import math
import tf as ttff
import cv_bridge
from sensor_msgs.msg import Image
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

#camara = baxter_interface.CameraController('head_camera')

#Iniciar Nodo

#rospy.init_node('Camara2', anonymous = True) #anonimo para que no tenga alcance de nombres
    
class Mueve:
        def __init__(self,arm,posiciones):

                # Brazo a utilizar
                self.limb           = arm
                self.limb_interface = baxter_interface.Limb(self.limb)

                # set speed as a ratio of maximum speed
                self.limb_interface.set_joint_position_speed(0.5)
                #self.other_limb_interface.set_joint_position_speed(0.5)

                # Gripper ("left" or "right")
                self.gripper = baxter_interface.Gripper(self.limb)

                # calibrate the gripper
                self.gripper.calibrate()

                # Laser
                self.laser = baxter_interface.AnalogIO('left_hand_range')
                self.dist = self.laser.state()/1000-0.105

                # Parametros de camara
                self.cam_calibracion = 0.0025            # 0.0025 pixeles por metro a 1 metro de distancia. Factor de correccion
                self.cam_x_offset    = 0              # Correccion de camara por los gripper,
                self.cam_y_offset    = -0.025       
                self.resolution      = 1
                self.width           = 1280               # 1280 640  960
                self.height          = 800               # 800  400  600

                # Pose inicial
                self.x = posiciones[0][0]   #0.4   # La tengo que setear
                self.y = posiciones[0][1]   #0.4   # La tengo que setear
                self.z = 0.0  
                self.roll = math.pi
                self.pitch = 0.0
                self.yaw = 0.0
                self.pose = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]
                self.mover_baxter('base',self.pose[:3],self.pose[3:6])

		# Guardar cuadrilatero de busqueda
		self.x_pos=self.x+abs(self.x-posiciones[4][0])
		self.x_neg=self.x-abs(self.x-posiciones[3][0])
		self.y_pos=self.y+abs(self.y-posiciones[1][1])
		self.y_neg=self.y-abs(self.y-posiciones[2][1])

		# Posiciones contenedores
		self.Contenedor_1=posiciones[5]
		self.Contenedor_2=posiciones[6]
		self.Contenedor_3=posiciones[7]
		self.Contenedor_4=posiciones[8]
		self.Contenedor_5=posiciones[9]
		

        def baxter_to_pixel(self, pt, dist):
                x = (self.width / 2) + int((pt[1] - (self.pose[1] + self.cam_y_offset)) / (self.cam_calibracion * dist))
                y = (self.height / 2) + int((pt[0] - (self.pose[0] + self.cam_x_offset)) / (self.cam_calibracion * dist))
                return (x, y)

        # convert image pixel to Baxter point
        def pixel_to_baxter(self, px, dist):
                x = ((px[1] - (self.height / 2)) * self.cam_calibracion * dist) + self.pose[0] + self.cam_x_offset
                y = ((px[0] - (self.width / 2)) * self.cam_calibracion * dist) + self.pose[1] + self.cam_y_offset
                return (x, y)

        def mensaje_matriz_a_pose(self,T, frame):
                t = PoseStamped()
                t.header.frame_id = frame
                t.header.stamp = rospy.Time.now()
                translacion = ttff.transformations.translation_from_matrix(T)
                orientacion = ttff.transformations.quaternion_from_matrix(T)
                t.pose.position.x = translacion[0]
                t.pose.position.y = translacion[1]
                t.pose.position.z = translacion[2]
                t.pose.orientation.x = orientacion[0]
                t.pose.orientation.y = orientacion[1]
                t.pose.orientation.z = orientacion[2]
                t.pose.orientation.w = orientacion[3]        
                return t

        def mover_baxter(self, source_frame, trans, rot):

                nombre_servicio = '/ExternalTools/' + self.limb + '/PositionKinematicsNode/IKService'

                # Creacion del servicio 
                servicio_ik = rospy.ServiceProxy(nombre_servicio,SolvePositionIK)

                # Frame
                frame = source_frame

                # Promedio de velocidad del brazo
                self.limb_interface.set_joint_position_speed(0.5)

                matrix = ttff.transformations.concatenate_matrices(ttff.transformations.translation_matrix(trans),
                    ttff.transformations.euler_matrix(rot[0],rot[1],rot[2])
                    )
                
                rospy.wait_for_service(nombre_servicio,10)

                ik_mensaje = SolvePositionIKRequest()

                ik_mensaje.pose_stamp.append(self.mensaje_matriz_a_pose(matrix, frame))

                try:
                        respuesta = servicio_ik(ik_mensaje)
                except:
                        print "Excepcion"

                if respuesta.isValid[0] == True:
                        movimiento =  dict(zip(respuesta.joints[0].name,respuesta.joints[0].position))
                        self.limb_interface.move_to_joint_positions(movimiento)
                        print "Movimiento Valido"
                else:
                        print "Movimiento no Valido"


	def randomm(self): 
		#self.x_pos,self.x_neg,self.y_pos,self.y_neg #Variables que tengo que usar
		aux_x=0
		aux_y=0
		# np.random.randint(2, size=1)[0]
		while 1:
			aux_x=self.pose[0]
			aux_x +=np.random.uniform(-0.08, 0.08, size=1)[0]
			if aux_x >= self.x_neg and aux_x <= self.x_pos:
				self.pose[0] = aux_x
				break
			else:
				continue
			
		while 1:
			aux_y=self.pose[1]
			aux_y +=np.random.uniform(-0.08, 0.08, size=1)[0]
			if aux_y >= self.y_neg and aux_y <= self.y_pos:
				self.pose[1] = aux_y
				break
			else:
				continue
			


        def CA(self,objectX,objectY,Herramienta,Angulo,Dimension):

                self.cx = objectX
                self.cy = objectY
		if (Dimension[0] < Dimension[1]) and (Angulo == -90 or Angulo == -0):
			self.angulo = -np.radians(Angulo)
		elif (Dimension[0] < Dimension[1]):
			self.angulo = -np.radians(Angulo)
		else:
			self.angulo = -np.radians(Angulo) + math.pi/2
         
                print [self.cx,self.cy]              

                (a,b) = self.pixel_to_baxter([self.cx,self.cy],0.30) #0.17

                self.mover_baxter('base',[a,b,self.pose[2]],[self.pose[3],self.pose[4],self.pose[5]+self.angulo])
		
	        self.mover_baxter('base',[a, b, self.pose[2] - 0.2],[self.pose[3], self.pose[4], self.pose[5]+self.angulo])

	        self.gripper.close()
	        
	        rospy.sleep(1)
	        
	        self.mover_baxter('base',[a,b,self.pose[2]],self.pose[3:6])

	        if self.gripper.force() != 0:

	                if Herramienta == 'Alicate':
	                        print 'Moviendo Alicate'
	                        self.mover_baxter('base',[self.Contenedor_2[0],self.Contenedor_2[1],self.pose[2]],self.pose[3:6])
				self.mover_baxter('base',[self.Contenedor_2[0],self.Contenedor_2[1],self.pose[2]- 0.1],self.pose[3:6])
				self.gripper.open()
	                        self.mover_baxter('base',[self.Contenedor_2[0],self.Contenedor_2[1],self.pose[2]],self.pose[3:6])

	                if Herramienta == 'Destornillador':	                        
	                        print 'Moviendo Destornillador'
	                        self.mover_baxter('base',[self.Contenedor_3[0],self.Contenedor_3[1],self.pose[2]],self.pose[3:6])
				self.mover_baxter('base',[self.Contenedor_3[0],self.Contenedor_3[1],self.pose[2]- 0.1],self.pose[3:6])
				self.gripper.open()
	                        self.mover_baxter('base',[self.Contenedor_3[0],self.Contenedor_3[1],self.pose[2]],self.pose[3:6])

	                if Herramienta == 'Martillo':	                        
	                        print 'Moviendo Martillo'
	                        self.mover_baxter('base',[self.Contenedor_1[0],self.Contenedor_1[1],self.pose[2]],self.pose[3:6])
				self.mover_baxter('base',[self.Contenedor_1[0],self.Contenedor_1[1],self.pose[2]- 0.1],self.pose[3:6])
				self.gripper.open()
	                        self.mover_baxter('base',[self.Contenedor_1[0],self.Contenedor_1[1],self.pose[2]],self.pose[3:6])

	                if Herramienta == 'Taladro':	                      
	                        print 'Moviendo Taladro'
	                        self.mover_baxter('base',[self.Contenedor_4[0],self.Contenedor_4[1],self.pose[2]],self.pose[3:6])
				self.mover_baxter('base',[self.Contenedor_4[0],self.Contenedor_4[1],self.pose[2]- 0.1],self.pose[3:6])
				self.gripper.open()
				self.mover_baxter('base',[self.Contenedor_4[0],self.Contenedor_4[1],self.pose[2]],self.pose[3:6])


			if Herramienta == 'Llave':	        
	                        print 'Moviendo Llave'
	                        self.mover_baxter('base',[self.Contenedor_5[0],self.Contenedor_5[1],self.pose[2]],self.pose[3:6])
				self.mover_baxter('base',[self.Contenedor_5[0],self.Contenedor_5[1],self.pose[2]- 0.1],self.pose[3:6])
				self.gripper.open()
				self.mover_baxter('base',[self.Contenedor_5[0],self.Contenedor_5[1],self.pose[2]],self.pose[3:6])

      			
	                

	                rospy.sleep(1)

	                self.mover_baxter('base',[self.pose[0],self.pose[1],self.pose[2]],self.pose[3:6])
	                print 'Movimiento completado!'

	        else:

	                self.gripper.open()

	                rospy.sleep(1)
	                self.mover_baxter('base',self.pose[:3],self.pose[3:6])
	                print 'Nada en el Gripper!'
