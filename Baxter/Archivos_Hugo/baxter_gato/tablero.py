#!/usr/bin/env 
# by Hugo Ubilla
import numpy as np

class tablero_gato():
    def __init__(self, ancho, largo, items_ancho, items_largo, altura = -0.05 ):
        '''
        ancho: Corresponde al ancho de la caja.
        largo: Corresponde al largo de la caja.
        items_ancho: Corresponde a la cantidad de cuadriculas de forma horizontal.
        items_largo: Corresponde a la cantidad de cuadriculas de forma vertical.
        altura: distancia que debe llegar el brazo en la parte superior

        Crear objeto y luego llamar a la funcion ubicacion(tran_x,trans_y) que corresponde
        a una translacion del centro superior izquierdo
        '''
        self.largo = largo * 1.0
        self.ancho = ancho * 1.0
        self.items_largo = items_largo
        self.items_ancho = items_ancho
        self.altura = altura
        self.espacios = items_largo * items_ancho
        self.calcular_centros()
        self.centros_en_origen()

    def calcular_centros(self):
        separacion_hor = (self.ancho / self.items_ancho)
        dist_borde_h = separacion_hor / 2.0

        separacion_ver = (self.largo / self.items_largo)
        dist_borde_v = separacion_ver / 2.0

        self.centros = []
        for j in range(self.items_largo):
            for i in range(self.items_ancho):
                self.centros.append([dist_borde_h + separacion_hor * j, dist_borde_v + separacion_ver * i]) 

    def centros_en_origen(self):
        '''
        Realiza una translacion y deja el centro superior izq en el origen del CF y adicion de la coordenada z
        Todo esta visto en funcion del base CF
        '''
        for i in range(len(self.centros)-1,-1,-1):
            self.centros[i][0] = self.centros[i][0] - self.centros[0][0]
            self.centros[i][1] = self.centros[i][1] - self.centros[0][1]

        for i in self.centros:
            i.append(self.altura)

    def ubicacion(self, trans_x, trans_y):
        '''
        trans_x : valor de x de la esquina superior derecha en la mesa
        trans_y : valor de y de la esquina superior derecha en la mesa 
        '''     
        for i in range(len(self.centros)):
            for j in range(len(self.centros[i])):
                if j == 0:
                    self.centros[i][j] = self.centros[i][j] + trans_x * 1.0
                if j == 1:
                    self.centros[i][j] = self.centros[i][j] + trans_y * 1.0


def main():
    caja1 = bandeja(0.4, 0.6, 2, 2, -0.2 )
    
    print caja1.centros

    caja1.centros_en_origen()

    print caja1.centros

    caja1.ubicacion(1.0,1.0)

    print caja1.centros


if __name__ == '__main__':
    main()