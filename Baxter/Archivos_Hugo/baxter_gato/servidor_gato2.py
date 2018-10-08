
import rospy

from std_msgs.msg import Int32, Int32MultiArray

from pantalla_gato import *

from configurador_gato import *

class gato():

    print ' G A T O '

    def __init__(self, turno):
        self.tablero = ['     ','|','     ','|','     ','\n','     ','|','     ','|','     ','\n','-----','|','-----','|','-----','\n','     ','|','     ','|','     ','\n','-----','|','-----','|','-----','\n','     ','|','     ','|','     ','\n','     ','|','     ','|','     ','\n',]
        self.tab_espacios = [6,8,10,18,20,22,30,32,34]
        self.tabla_inicial = range(9)
        self.turno = turno  # 1 o 2
        self.movimientos = []
        self.sub_movimiento = rospy.Subscriber('mov', Int32, self.obtener_movimiento)
        self.publicador = rospy.Publisher('turno', Int32, queue_size = 5)
        self.ganador = rospy.Publisher('ganador', Int32, queue_size = 5)

        # Iniciar pantalla
        self.pantalla_gato = pantalla_gato()

        # Inicial brazos
        self.mov_left = configurador_gato('left', 0.03)
        self.mov_right = configurador_gato('right', 0.03)

        # Entrega de posiciones iniciales y finales.
        self.mov_left.pos_ini = [0.6213194238378937, 0.292417612494151, -0.20415115100944747, -3.133947270636099, -0.011662805419859626, -1.4994276754150357]
        self.mov_left.pos_fin = [0.3803126110241114, -0.16323111726976505, -0.1727197791797792, 3.0918552376332546, 0.023658354174193257, -1.5294580736608432]
        self.mov_right.pos_ini = [0.6646422385349072, -0.3405367908663159, -0.21056248943235825, -3.1017759821314037, -0.06399182689688478, -1.5229167389621114]
        self.mov_right.pos_fin =[0.37814423775055694, -0.1751383569787016, -0.18244245831615022, -3.125071365646154, -0.036023901202946886, -1.5086309690181061]

        self.mov_left.definir_posicion(True)
        self.mov_right.definir_posicion(True)
        
        self.mov = None

    def jugada(self, jugador):

        while not rospy.is_shutdown():
            if self.mov not in self.movimientos and self.mov in range(9):
                self.movimientos.append(self.mov)
                print jugador
                self.pantalla_gato.hacer_jugada(jugador, self.mov)
                
                if jugador == 1:
                    self.mov_left.mover_ini_fin_posicion(self.mov)
                else:
                    self.mov_right.mover_ini_fin_posicion(self.mov)
                
                if jugador == 1:
                    self.tablero[self.tab_espacios[self.mov]] = '  X  '
                    self.tabla_inicial[self.mov] = 'X'
                else:
                    self.tablero[self.tab_espacios[self.mov]] = '  O  '
                    self.tabla_inicial[self.mov] = 'O'

                self.visualizar_gato()              
                break
            else:
                rospy.sleep(0.5)
                if jugador == 1:
                    self.publicador.publish(1)
                else:
                    self.publicador.publish(2)


    def obtener_movimiento(self, datos):
        self.mov = datos.data

    def obtener_turno(self, datos):
        self.turno = datos.data 

    def verificar_gato(self,x):
        if ((x[0] == x[1] and x[1] == x[2]) or
            (x[3] == x[4] and x[4] == x[5]) or
            (x[6] == x[7] and x[7] == x[8]) or
            (x[0] == x[3] and x[3] == x[6]) or
            (x[1] == x[4] and x[4] == x[7]) or
            (x[2] == x[5] and x[5] == x[8]) or
            (x[0] == x[4] and x[4] == x[8]) or
            (x[2] == x[4] and x[4] == x[6])):
            return True
        return False

    def visualizar_gato(self):
        for i in self.tablero:
            print i,

    def jugar(self):
        for i in range(9):
            self.jugada(self.turno)
            if self.verificar_gato(self.tabla_inicial) == True :
                print "Felicidades jugador ", self.turno, "!!!!"
                self.pantalla_gato.resultado(self.turno)
                if self.turno == 1:
                    self.mov_left.ganador()
                else:
                    self.mov_right.ganador()
                break
            if self.turno == 1:
                self.turno = 2
            else:
                self.turno = 1
            if i == 8:
                print "Empate !"
                self.pantalla_gato.resultado(0)
        self.ganador.publish(1)

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

hola = gato(jugador_1)
hola.jugar()


