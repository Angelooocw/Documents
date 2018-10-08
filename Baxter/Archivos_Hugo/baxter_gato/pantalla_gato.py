#!/usr/bin/env 

import rospy
import cv2
import cv_bridge
from sensor_msgs.msg import Image
import os
import numpy as np


class pantalla_gato():
    def __init__(self):
        home = os.getenv('HOME')  #

        # Editar carpeta
        home = home + '/ros_ws/gato'

        # Imagenes
        self.p_principal = cv2.imread(home+'/images/Gato.jpg')
        self.circulo = cv2.imread(home+'/images/Circulo.jpg')
        self.cruz = cv2.imread(home+'/images/Cruz.jpg')
        self.ganador_x = cv2.imread(home+'/images/GanadorX.jpg')
        self.ganador_o = cv2.imread(home+'/images/GanadorO.jpg')
        self.empate = cv2.imread(home+'/images/Empate.jpg')

        # Dimensiones

        self.cir_altura_y = len(self.circulo)
        self.cir_ancho_x = len(self.circulo[0])

        self.gan_altura_y = len(self.ganador_x)
        self.gan_ancho_x = len(self.ganador_x[0])

        # Jugadas

        x1,x2,x3 = 440,600,760
        y1,y2,y3 = 100,250,400

        self.pts = [[y1,x1],[y1,x2],[y1,x3],[y2,x1],[y2,x2],[y2,x3],[y3,x1],[y3,x2],[y3,x3]]

        # Publicador
        self.pub = rospy.Publisher('/robot/xdisplay', Image, latch = True , queue_size = 10)

        self.publicador_a_pantalla()

        # Los pixeles estan ordenados de [y_hacia_abajo, x_positivo] o de otra forma,
        # es un arreglo donde cada valor es una horizontal de la foto

    def publicador_a_pantalla(self):
        msg = cv_bridge.CvBridge().cv2_to_imgmsg(self.p_principal)
        self.pub.publish(msg)
        rospy.sleep(1)

    def hacer_jugada(self, jugador, indice_jugada):
        '''
        indice_jugada: corresponde a un numero del 0 al 8 que permite colocar un valor en el tablero
        jugador: correspone a 1 o 2 para ver si la jugada es de O o X
        '''
        if jugador == 1:
            self.p_principal[self.pts[indice_jugada][0]:self.pts[indice_jugada][0]+self.cir_altura_y,self.pts[indice_jugada][1]:self.pts[indice_jugada][1]+self.cir_ancho_x] = self.cruz
        else:
            self.p_principal[self.pts[indice_jugada][0]:self.pts[indice_jugada][0]+self.cir_altura_y,self.pts[indice_jugada][1]:self.pts[indice_jugada][1]+self.cir_ancho_x] = self.circulo

        self.publicador_a_pantalla()


    def resultado(self, resultado):
        '''
        resultado: 0,1,2 donde 1 gana jugador 1, 2 gana jugador 2 y 0 empate
        '''
        x4 = 40
        y4 = 400 
        if resultado == 1:
            self.p_principal[y4:y4+self.gan_altura_y,x4:x4+self.gan_ancho_x] = self.ganador_x
        elif resultado == 2:
            self.p_principal[y4:y4+self.gan_altura_y,x4:x4+self.gan_ancho_x] = self.ganador_o
        else:
            self.p_principal[y4:y4+self.gan_altura_y,x4:x4+self.gan_ancho_x] = self.empate
        self.publicador_a_pantalla()


def main():
    rospy.init_node('ffdsdsfads', anonymous = True)
    gato = pantalla_gato()
    gato.hacer_jugada(0,0)    

if __name__ == '__main__':
    main()