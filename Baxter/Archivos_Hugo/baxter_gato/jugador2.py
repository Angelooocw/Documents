#!/usr/bin/env 
# by Hugo Ubilla
# Universidad de Concepcion 

import rospy
from std_msgs.msg import Int32
from termios import tcflush, TCIFLUSH
import sys 

rospy.init_node("jugador2", anonymous = True)

x = None
gan = None
def obtener_turno(datos):
    global x
    x = datos.data 
    #print x

def obtener_ganador(datos):
    global gan
    gan = datos.data

rospy.Subscriber('turno', Int32, obtener_turno)
rospy.Subscriber('ganador', Int32, obtener_ganador)
#pub_turno = rospy.Publisher('turno', Int32, queue_size = 5)
pub_movimiento = rospy.Publisher('mov', Int32, queue_size = 5)

while not rospy.is_shutdown():
    if gan == 1:
        break
    if x == 2:
        while True:
            print "Jugador 2:"
            print "Ingrese jugada del 1 al 9 ...:",
            tcflush(sys.stdin, TCIFLUSH)
            mov = raw_input()

            try:
                mov = int(mov) - 1
                if mov > 8 or mov < 0:
                    print "Ingrese un valor valido entre 0 y 8"
                    continue
                else:
                    pub_movimiento.publish(mov)
                    print "Espere..."
                    x = None
                    rospy.sleep(2)
                    x = 40
                    break
            except:
                print "Ingrese un valor numerico"





