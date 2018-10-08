#!/usr/bin/env 
# by Hugo Ubilla
# Universidad de Concepcion 

import rospy

from std_msgs.msg import Int32, Int32MultiArray

rospy.init_node("nodo_0_1")

publicador = rospy.Publisher('turno', Int32, queue_size = 5)

while not rospy.is_shutdown():
    print "Seleccione que jugador comienza (1 o 2)"
    jugador_1 = raw_input()
    try:
        jugador_1 = int(jugador_1)
        if jugador_1 < 1 or jugador_1 > 2:
            continue
        break
    except:
        continue

publicador.publish(jugador_1)

print "El jugador ", jugador_1, " comienza el juego." 

rospy.spin()