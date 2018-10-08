
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
        self.mov_left = configurador_gato('left')
        self.mov_right = configurador_gato('right')

        # Posiciones iniciales y finales.
	    ########## REEMPLAZAR AQUI ##########
     
        self.mov_left.pos_ini = [0.5665774439950692, 0.428117174975678, -0.2092624182305786, 3.1257665698068764, 0.10406832660891595, -1.5371161939134355]
        self.mov_left.pos_fin = [0.45693656155412693, -0.11428189837371774, -0.20750769005127004, -3.108360442295319, 0.03611597202836131, -1.4954946777390992]
        self.mov_right.pos_ini = [0.5663902709804273, -0.41955931072379565, -0.2069010872532103, 3.1102578875431113, -0.09339455405332188, -1.6228706525168715]
        self.mov_right.pos_fin =[0.453481833420202, -0.11904171783819699, -0.1991129994825413, -3.1375637616251315, -0.025242863877158656, -1.597195735665272]

        #####################################

        self.mov_left.definir_posicion(True)
        self.mov_right.definir_posicion(True)
        
        self.mov = None

    def jugada(self, jugador):

        while not rospy.is_shutdown():
            if self.mov not in self.movimientos and self.mov in range(9):
                self.movimientos.append(self.mov)
                print jugador
                if jugador == 1:
                    self.mov_left.mover_ini_fin_posicion(self.mov)
                else:
                    self.mov_right.mover_ini_fin_posicion(self.mov)
                self.pantalla_gato.hacer_jugada(jugador, self.mov)
                break
            else:
                if jugador == 1:
                    self.publicador.publish(1)
                else:
                    self.publicador.publish(2)
            

        if jugador == 1:
            self.tablero[self.tab_espacios[self.mov]] = '  X  '
            self.tabla_inicial[self.mov] = 'X'
        else:
            self.tablero[self.tab_espacios[self.mov]] = '  O  '
            self.tabla_inicial[self.mov] = 'O'

        self.visualizar_gato()

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
                    print "gano jugador 1"
                else:
                    self.mov_right.ganador()
                    print "gano jugador 2"
                break
            if self.turno == 1:
                self.turno = 2
            else:
                self.turno = 1
            if i == 8:
                print "Empate !"
                self.pantalla_gato.resultado(0)
        self.ganador.publish(1)


def main():

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

if __name__=='__main__':
    main()
