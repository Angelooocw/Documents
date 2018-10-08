#!/usr/bin/env 

import baxter_interface
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
import tf
import math
from sensor_msgs.msg import Image
import numpy as np
import cv2 
import cv_bridge

class movimiento_cajas():
    def __init__(self, arm):
        self.arm = arm
        self.brazo = baxter_interface.Limb(self.arm)

        # Declaracion del boton
        self.boton_brazo = baxter_interface.Navigator(self.arm)

        # Gripper
        self.gripper = baxter_interface.Gripper(self.arm)
        
        # Calibracion
        self.gripper.calibrate()

        self.pos_caja = [1] * 6 
        self.pos_caja_vacia = [1] * 6 
        self.pos_caja_llena = [1] * 6 

        # Rotacion estandar
        self.rot = [math.pi, 0, -math.pi/2]

        # Publicador a pantalla
        self.pub = rospy.Publisher('/robot/xdisplay', Image, latch = True , queue_size = 10)

        # Datos de cajas
        self.num_cajas = 2
        self.altura_cajas = 0.1
        self.nivel = 0

        self.contador = [False] * 6

        self.lugar = 0


    def publicador_a_pantalla(self,texto):
        position = (50, 300)
        fondo_negro = np.full((600,1024, 3), 0, dtype = np.uint8)
        cv2.putText(fondo_negro, texto, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 4)
        msg = cv_bridge.CvBridge().cv2_to_imgmsg(fondo_negro)
        self.pub.publish(msg)
        #rospy.sleep(1)

    def endpoint_a_pose_stamped(self, baxter_endpoint, frame):
        '''
        Transforma un msg del tipo endpoint_pose definido por la API de baxter_interface
        en un msg PoseStamped
        '''
        pos = PoseStamped()
        pos.header.stamp = rospy.Time.now()
        pos.header.frame_id = frame
        pos.pose.position.x = baxter_endpoint['position'][0]
        pos.pose.position.y = baxter_endpoint['position'][1]
        pos.pose.position.z = baxter_endpoint['position'][2]
        pos.pose.orientation.x = baxter_endpoint['orientation'][0]
        pos.pose.orientation.y = baxter_endpoint['orientation'][1]
        pos.pose.orientation.z = baxter_endpoint['orientation'][2]
        pos.pose.orientation.w = baxter_endpoint['orientation'][3]
        return pos

    def endpoint_a_vector_trans_euler(self, baxter_endpoint):
        '''
        Pasa de endpoint a vector de dimension 6 con 3 componentes de translacion y 3 de rotacion (euler)
        '''
        pos = []
        pos.append(baxter_endpoint['position'][0])
        pos.append(baxter_endpoint['position'][1])
        pos.append(baxter_endpoint['position'][2])
        euler = tf.transformations.euler_from_quaternion(baxter_endpoint['orientation'])
        pos.append(euler[0])
        pos.append(euler[1])
        pos.append(euler[2])
        return pos

    def matriz_a_pose_stamped(self, T, frame):
        '''
        Transforma una matriz homogenenea en un msg del tipo pose_stamped
        '''
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
    
    def definir_posicion(self):
        '''
        Guarda las posiciones iniciales y finales.
        '''
        self.publicador_a_pantalla("Presione el boton de navegacion para guardar posicion de la caja")
        while True:
            if self.boton_brazo.button0 == 1:
                self.publicador_a_pantalla("Guardando...")
                self.pos_caja = self.endpoint_a_vector_trans_euler(self.brazo.endpoint_pose())
                if self.tomar_dejar_caja() == True:
                    self.publicador_a_pantalla('Posicion correcta 1')
                    print 'Posicion correcta 1'
                    rospy.sleep(1)
                    break
                else:
                    self.publicador_a_pantalla('Inserte otra vez la posicion de la caja...')
                    print 'Inserte otra vez la posicion de la caja...'
            if self.boton_brazo.button1 == 1:
                self.publicador_a_pantalla('Saliendo...')
                rospy.sleep(1)
                break

        self.publicador_a_pantalla("Presione el boton de navegacion para guardar posicion de las cajas vacias")
        while True:
            if self.boton_brazo.button0 == 1:
                self.publicador_a_pantalla("Guardando...")
                self.pos_caja_vacia = self.endpoint_a_vector_trans_euler(self.brazo.endpoint_pose())
                if self.tomar_caja_vacia() == True:
                    self.publicador_a_pantalla('Posicion correcta caja vacia')
                    print 'Posicion correcta caja vacia'
                    rospy.sleep(1)
                    break
                else:
                    self.publicador_a_pantalla('Inserte otra vez la posicion de la caja vacia...')
                    print 'Inserte otra vez la posicion de las cajas vacias...'
            if self.boton_brazo.button1 == 1:
                self.publicador_a_pantalla('Saliendo...')
                rospy.sleep(1)
                break

        self.publicador_a_pantalla("Presione el boton de navegacion para guardar posicion de las cajas llenas")
        while True:
            if self.boton_brazo.button0 == 1:
                self.publicador_a_pantalla("Guardando...")
                self.pos_caja_llena = self.endpoint_a_vector_trans_euler(self.brazo.endpoint_pose())
                if self.dejar_caja_llena() == True:
                    self.publicador_a_pantalla('Posicion correcta caja llenas')
                    print 'Posicion correcta caja llena'
                    rospy.sleep(1)
                    break
                else:
                    self.publicador_a_pantalla('Inserte otra vez la posicion de la cajas llenas...')
                    print 'Inserte otra vez la posicion de las cajas llenas...'               
            if self.boton_brazo.button1 == 1:
                self.publicador_a_pantalla("Saliendo...")
                rospy.sleep(1)
                break


    def mover_baxter(self, source_frame, trans, rot, ejecutar):
        
        nombre_servicio = '/ExternalTools/'+self.arm+'/PositionKinematicsNode/IKService'    
        servicio_ik = rospy.ServiceProxy(nombre_servicio,SolvePositionIK)
        frame = source_frame   

        # Promedio de velocidad del brazo
        self.brazo.set_joint_position_speed(0.5)

        matrix = tf.transformations.concatenate_matrices(tf.transformations.translation_matrix(trans),
            tf.transformations.euler_matrix(rot[0],rot[1],rot[2])
            )
        
        rospy.wait_for_service(nombre_servicio,10)  
        ik_mensaje = SolvePositionIKRequest()
        ik_mensaje.pose_stamp.append(self.matriz_a_pose_stamped(matrix, frame))

        try:
            respuesta = servicio_ik(ik_mensaje)
        except:
            ejec = False
            #print "Movimiento no ejecutado 1"

        #print respuesta.isValid[0]

        ejec = respuesta.isValid[0]

        if ejec == True:
            movimiento =  dict(zip(respuesta.joints[0].name,respuesta.joints[0].position))
            if ejecutar == True:
                self.brazo.move_to_joint_positions(movimiento)
        #else:
            #print "Movimiento no ejecutado 2"

        return ejec

    def tomar_caja(self, vector_pose, ejecutar):
        '''
        Funcion que baja hacia una caja, la toma y luego la sube.
        ejecutar: True si se requiere que se ejecute el movimiento y False si solo se requiere verificar si se puede realizar.
        '''
        v1 = self.mover_baxter('base',[vector_pose[0], vector_pose[1], vector_pose[2] + 0.20], self.rot, ejecutar)
        v2 = self.mover_baxter('base',vector_pose[:3], self.rot, ejecutar)
        if ejecutar == True:
            self.gripper.close()
            rospy.sleep(1)
        v3 = self.mover_baxter('base',[vector_pose[0], vector_pose[1], vector_pose[2] + 0.20], self.rot, ejecutar)
        if self.gripper.force() != 0:
            self.contador[self.lugar] = True
        if v1 == False or v2 == False or v3 == False: 
            return False
        return True

    def dejar_caja(self, vector_pose, ejecutar):
        '''
        Funcion que deja la caja llenas en otra posicion, la suelta y luego sube el brazo.
        ejecutar: True si se requiere que se ejecute el movimiento y False si solo se requiere verificar si se puede realizar.
        '''
        v1 = self.mover_baxter('base',[vector_pose[0], vector_pose[1], vector_pose[2] + 0.20], self.rot, ejecutar)
        v2 = self.mover_baxter('base',vector_pose[:3], self.rot, ejecutar)
        if ejecutar == True:
            self.gripper.open()
            rospy.sleep(1)
        v3 = self.mover_baxter('base',[vector_pose[0], vector_pose[1], vector_pose[2] + 0.20], self.rot, ejecutar)
        if v1 == False or v2 == False or v3 == False: 
            return False
        return True

    def tomar_dejar_caja(self):
        v = self.tomar_caja(self.pos_caja, False)
        if v == False:
            return False
        return True

    def dejar_caja_llena(self):
        for i in range(self.num_cajas):
            v = self.dejar_caja(self.pos_caja_llena[:2] + [self.pos_caja_llena[2] + i * self.altura_cajas] + self.pos_caja_llena[3:6], False)
            if v == False:
                return False
        return True      

    def tomar_caja_vacia(self):
        for i in range(self.num_cajas):
            v = self.tomar_caja(self.pos_caja_vacia[:2] + [self.pos_caja_vacia[2] - i * self.altura_cajas] + self.pos_caja_vacia[3:6], False)
            if v == False:
                return False
        return True  


    def ejecutar_movimiento(self):
        self.tomar_caja(self.pos_caja, True)
        self.dejar_caja(self.pos_caja_llena[:2] + [self.pos_caja_llena[2] + self.nivel * self.altura_cajas] + self.pos_caja_llena[3:6], True)
        self.tomar_caja(self.pos_caja_vacia[:2] + [self.pos_caja_vacia[2] - self.nivel * self.altura_cajas] + self.pos_caja_vacia[3:6], True)
        self.dejar_caja(self.pos_caja, True)
        self.nivel = self.nivel + self.altura_cajas

def main():
    rospy.init_node('holasdfa', anonymous = True)
    mov_caja = movimiento_cajas('right')
    mov_caja.definir_posicion()
    mov_caja.ejecutar_movimiento()



if __name__=='__main__':
    main()