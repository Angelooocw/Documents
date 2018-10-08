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
        self.gripper.calibrate()

        # Declaracion del brazo 
        self.brazo = baxter_interface.Limb(self.arm)

        self.pos_ini = [1] * 6
        self.pos_fin = [1] * 6

        # Largo de la pieza
        self.separacion = 0.0315

        # Tablero
        self.caja = tablero_gato(0.38,0.38,3,3)

        self.centros = [x[:] for x in self.caja.centros]

        # Rotacion estandar
        self.rot = [math.pi, 0, -math.pi/2]

        # Publicador a pantalla
        self.pub = rospy.Publisher('/robot/xdisplay', Image, latch = True , queue_size = 10)

        self.fila_max = 5

        self.fila = 0

        self.contador = [False] * self.caja.espacios

        self.lugar = 0

        #self.publicador_a_pantalla('Simulador Produccion')


    def publicador_a_pantalla(self,texto):
        position = (10, 300)
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


    def definir_posicion(self, isPosition):
        '''
        Guarda las posiciones iniciales y finales.
        '''
        if isPosition == False:
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

            print "Presione el boton de navegacion para guardar posicion esquina gato"
            self.publicador_a_pantalla("Presione el boton de navegacion \n para guardar posicion esquina gato")
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
        if isPosition == True:
            # Asignar pos_fin con anterioridad antes de llamavar a la funcion
            self.caja.ubicacion(self.pos_fin[0], self.pos_fin[1])       

    def mover_baxter(self, source_frame, trans, rot, ejecutar):
        
        nombre_servicio = '/ExternalTools/'+self.arm+'/PositionKinematicsNode/IKService'    
        servicio_ik = rospy.ServiceProxy(nombre_servicio,SolvePositionIK)
        frame = source_frame   

        # Promedio de velocidad del brazo
        self.brazo.set_joint_position_speed(1)

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
            if self.arm == 'left':
                valor = self.tomar([self.pos_ini[0], self.pos_ini[1] + self.separacion * i] + self.pos_ini[2:6], False)
            else:
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
            #valor = self.soltar([i[0], i[1], self.pos_fin[2], self.rot[0], self.rot[1], self.rot[2]], False)
            print valor
            print i
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
            rospy.sleep(0.5)
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


    def mover_ini_fin_posicion(self, posicion):
        '''
        Funcion que ejecuta todo el algoritmo
        '''
        if self.arm == 'left':
            self.tomar([self.pos_ini[0], self.pos_ini[1] + self.separacion * self.fila] + self.pos_ini[2:6], True)
        else:
            self.tomar([self.pos_ini[0], self.pos_ini[1] - self.separacion * self.fila] + self.pos_ini[2:6], True)
        self.soltar(self.caja.centros[posicion][:2] + self.pos_fin[2:6], True)
        self.fila = self.fila + 1
        self.lugar = posicion

        # Posicion de espera
        self.mover_baxter('base',[self.pos_ini[0], self.pos_ini[1], self.pos_ini[2] + 0.20], self.rot, True)

    def ganador(self):
        if self.arm =='right':
            x_ini       = 0.31
            y_ini       = -0.62
            z_ini       = 1.4
            roll_ini    = 0 
            pitch_ini   = 0 
            yaw_ini     = 0 

            self.mover_baxter('base',[x_ini, y_ini, z_ini],[roll_ini, pitch_ini, yaw_ini],True)
            self.mover_baxter('base',[x_ini, y_ini, z_ini - 0.3],[roll_ini, pitch_ini, yaw_ini],True)
            self.mover_baxter('base',[x_ini, y_ini, z_ini],[roll_ini, pitch_ini, yaw_ini],True)

        else:
            x_ini       = 0.31
            y_ini       = 0.62
            z_ini       = 1.4
            roll_ini    = 0
            pitch_ini   = 0 
            yaw_ini     = 0 

            self.mover_baxter('base',[x_ini, y_ini, z_ini],[roll_ini, pitch_ini, yaw_ini],True)
            self.mover_baxter('base',[x_ini, y_ini, z_ini - 0.3],[roll_ini, pitch_ini, yaw_ini],True)
            self.mover_baxter('base',[x_ini, y_ini, z_ini],[roll_ini, pitch_ini, yaw_ini],True)



def main():
    rospy.init_node("boton_ini")
    pos_left = configurador_gato('left')
    pos_right = configurador_gato('right')
    
    # Seccion para crear nuevas posiciones

    pos_left.definir_posicion(False)  # False: No hay posicion y permite crear una nueva
    pos_right.definir_posicion(False) # False: No hay posicion y permite crear una nueva

    posiciones = open("posiciones.py","w")
    posiciones.write('self.mov_left.pos_ini = '+str(pos_left.pos_ini)+'\n')
    posiciones.write('self.mov_left.pos_fin = '+str(pos_left.pos_fin)+'\n')
    posiciones.write('self.mov_right.pos_ini = '+str(pos_right.pos_ini)+'\n')
    posiciones.write('self.mov_right.pos_fin ='+str( pos_right.pos_fin)+'\n')
    posiciones.close()
    

    '''
    pos_left.pos_ini = [0.6248477316599359, 0.288532683357022, -0.21567631424436856, 3.09246970217757, 0.0016382263897491687, -1.5266920782613682]
    pos_left.pos_fin = [0.6295677457579234, -0.1332612681722616, -0.13011934814131154, 3.126778426366641, -0.013580935352520383, -1.5634450751835467]
    pos_right.pos_ini = [0.6701092921478491, -0.40089024846318333, -0.20468270633627295, 3.1221108557083945, -0.05417571775553071, -1.527537798314209]
    pos_right.pos_fin =[0.6265570410016598, -0.13388788913183888, -0.13835421508952161, 3.0740439733460283, -0.03320794609997299, -1.598962388235756]
    pos_left.definir_posicion(True)    
    pos_right.definir_posicion(True)
    
    pos_left.ganador()
    pos_right.ganador()
    '''
    
    '''
    # Seccion para hacer pruebas
    #pos_left.mover_ini_fin_posicion(0)
    #pos_right.mover_ini_fin_posicion(1)
    #pos_left.mover_ini_fin_posicion(2)
    #pos_right.mover_ini_fin_posicion(3)
    #pos_left.mover_ini_fin_posicion(4)
    #pos_right.mover_ini_fin_posicion(5)
    #pos_left.mover_ini_fin_posicion(6)
    #pos_right.mover_ini_fin_posicion(7)
    #pos_left.mover_ini_fin_posicion(8)
    
    '''

if __name__=='__main__':
    main()
