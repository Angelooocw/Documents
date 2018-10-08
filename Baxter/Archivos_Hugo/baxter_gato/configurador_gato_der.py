#!/usr/bin/env 

import baxter_interface
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
import tf
from tablero import *
import math
from sensor_msgs.msg import Image
import cv2
import numpy as np
import cv_bridge

class configurador_gato():
    def __init__(self, arm, separacion):
        
        self.arm = arm

        # Declaracion del boton
        self.boton_brazo = baxter_interface.Navigator(self.arm)

        # Gripper
        self.gripper = baxter_interface.Gripper(self.arm)
        
        # Calibracion
        #self.gripper.calibrate()

        # Declaracion del brazo 
        self.brazo = baxter_interface.Limb(self.arm)

        self.pos_ini = [1] * 6
        self.pos_fin = [1] * 6

        # Declaracion de caja
        self.caja = tablero_gato(0.3,0.3,3,3)
        self.separacion = separacion

        self.centros = [x[:] for x in self.caja.centros]

        # Rotacion estandar
        self.rot = [math.pi, 0, -math.pi/2]

        # Publicador a pantalla
        self.pub = rospy.Publisher('/robot/xdisplay', Image, latch = True , queue_size = 10)

        self.contador = [False] * self.caja.espacios

        self.lugar = 0

        self.fila_max = 5

        self.fila = 0

        self.definir_posicion()

        self.publicador_a_pantalla('Simulador Produccion')

    def publicador_a_pantalla(self,texto):
        position = (300, 300)
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
        print "Presione el boton de navegacion para guardar posicion inicial de la fila"
        self.publicador_a_pantalla("Presione el boton de navegacion para guardar posicion inicial de la fila")
        while True:
            if self.boton_brazo.button0 == 1:
                print "Guardando..."
                self.pos_ini = self.endpoint_a_vector_trans_euler(self.brazo.endpoint_pose())
                if self.mover_ini() == True:
                    self.publicador_a_pantalla('Posicion correcta 1')
                    print 'Posicion correcta 1'
                    rospy.sleep(1)
                    break
                else:
                    self.publicador_a_pantalla('Inserte otra vez 1...')
                    print 'Inserte otra vez 1...'
            if self.boton_brazo.button1 == 1:
                self.publicador_a_pantalla('Saliendo...')
                rospy.sleep(1)
                break
    
        print "Presione el boton de navegacion para guardar posicion final"
        self.publicador_a_pantalla("Presione el boton de navegacion para guardar posicion final")
        while True:
            if self.boton_brazo.button0 == 1:
                print "Guardando..."
                self.pos_fin = self.endpoint_a_vector_trans_euler(self.brazo.endpoint_pose())
                self.caja.ubicacion(self.pos_fin[0], self.pos_fin[1])
                if self.mover_fin() == True:
                    self.publicador_a_pantalla('Posicion correcta del primer lugar de la caja')
                    print 'Posicion correcta 2'
                    rospy.sleep(1)
                    break
                else:
                    self.caja.centros = [x[:] for x in self.centros]
                    self.publicador_a_pantalla('Inserte otra vez el primer lugar de la caja...')
                    print 'Inserte otra vez 2...'
            if self.boton_brazo.button1 == 1:
                self.publicador_a_pantalla('Saliendo...')
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

    def mover_ini(self):
        '''
        Funcion que retorna true si es posible realizar todos los movimientos que requieren tomar piezas
        '''
        for i in range(len(self.caja.centros)):
            # separacion negativa para el brazo derecho
            valor = self.tomar([self.pos_ini[0], self.pos_ini[1] - self.separacion * i] + self.pos_ini[2:6], False)
            if valor == False:
                return False
                break
        return True

    def mover_fin(self):
        '''
        Funcion que retorna true si es posible realizar todos los movimientos que requieren soltar piezas
        '''
        for i in self.caja.centros:
            valor = self.soltar(i[:2] + self.pos_fin[2:6], False)
            if valor == False:
                return False
                break
        return True

    def tomar(self, vector_pose, ejecutar):
        '''
        Funcion que baja hacia una pieza, la toma y luego la sube.
        ejecutar: True si se requiere que se ejecute el movimiento y False si solo se requiere verificar si se puede realizar.
        '''
        v1 = self.mover_baxter('base',[vector_pose[0], vector_pose[1], vector_pose[2] + 0.20], self.rot, ejecutar)
        v2 = self.mover_baxter('base',vector_pose[:3], self.rot, ejecutar)
        if ejecutar == True:
            self.gripper.close()
        v3 = self.mover_baxter('base',[vector_pose[0], vector_pose[1], vector_pose[2] + 0.20], self.rot, ejecutar)
        if self.gripper.force() != 0:
            self.contador[self.lugar] = True
        if v1 == False or v2 == False or v3 == False: 
            return False
        return True

    def soltar(self, vector_pose, ejecutar):
        '''
        Funcion que deja la pieza en una parte del tablero, la suelta y luego sube el brazo.
        ejecutar: True si se requiere que se ejecute el movimiento y False si solo se requiere verificar si se puede realizar.
        '''
        v1 = self.mover_baxter('base',[vector_pose[0], vector_pose[1], vector_pose[2] + 0.20], self.rot, ejecutar)
        v2 = self.mover_baxter('base',vector_pose[:3], self.rot, ejecutar)
        if ejecutar == True:
            self.gripper.open()
        v3 = self.mover_baxter('base',[vector_pose[0], vector_pose[1], vector_pose[2] + 0.20], self.rot, ejecutar)
        if v1 == False or v2 == False or v3 == False: 
            return False
        return True

    '''

    def mover_ini_fin(self):
        '''
        #Funcion que ejecuta todo el algoritmo
        '''
        while not rospy.is_shutdown():
            while self.lugar < self.caja.espacios:
                self.tomar([self.pos_ini[0], self.pos_ini[1] + self.separacion * self.fila] + self.pos_ini[2:6], True)
                if self.contador[self.lugar] == True:
                    self.soltar(self.caja.centros[self.lugar][:2] + self.pos_fin[2:6], True)
                    self.lugar = self.lugar + 1
                    self.fila = self.fila + 1
                else:
                    self.gripper.open()
                if self.contador == [True] * self.caja.espacios: 
                    self.contador = [False] * self.caja.espacios
                    self.lugar = 0
                    # Llamar a cambio de caja
                    self.mov_caja.ejecutar_movimiento()
                if self.fila == self.fila_max:
                    self.fila = 0
            break
    '''


def main():
    rospy.init_node("boton_ini")
    pos = configurador_gato('left', 0.03)
    #pos.mover_ini_fin()

if __name__=='__main__':
    main()
