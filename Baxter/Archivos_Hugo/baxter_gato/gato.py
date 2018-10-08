#!/usr/bin/env 
# by Hugo Ubilla
# Universidad de Concepcion 

class gato():

    print ' G A T O '

    def __init__(self):
        self.tablero = ['     ','|','     ','|','     ','\n','     ','|','     ','|','     ','\n','-----','|','-----','|','-----','\n','     ','|','     ','|','     ','\n','-----','|','-----','|','-----','\n','     ','|','     ','|','     ','\n','     ','|','     ','|','     ','\n',]
        self.tab_espacios = [6,8,10,18,20,22,30,32,34]
        self.tabla_inicial = range(9)
        self.turno = 0  # 0 o 1
        self.movimientos = []

    def jugada(self,jugador):

        while True:
            print "Jugador ", jugador, ":"
            print "Ingrese jugada del 0 al 8 \n"
            mov = raw_input()

            try:
                mov = int(mov)
                if mov in self.movimientos or mov > 8 or mov < 0:
                    print "Ingrese un valor valido entre 0 y 8"
                    continue
                else:
                    break
            except:
                print "Ingrese un valor numerico"

        self.movimientos.append(mov)

        if jugador == 0:
            self.tablero[self.tab_espacios[mov]] = '  X  '
            self.tabla_inicial[mov] = 'X'
        else:
            self.tablero[self.tab_espacios[mov]] = '  O  '
            self.tabla_inicial[mov] = 'O'

        self.visualizar_gato()

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
                break
            if self.turno == 0:
                self.turno = 1
            else:
                self.turno = 0
            if i == 8:
                print "Empate !"

hola = gato()

hola.jugar()