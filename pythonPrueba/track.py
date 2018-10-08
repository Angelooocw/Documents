import cv2
import numpy as np

cam = cv2.VideoCapture(0)
#kernel matriz para filtrar
kernel = np.ones((5,5),np.uint8)

while(True):
	#se obtiene las imagenes de la camara
	#read() devuelve una tupla que es almacenada en las variables ret y frame
    ret,frame = cam.read()
    #rangomaximo y minimo para detectar un color, color verde
    # np.array([B, G, R])
    rangomax=np.array([50,50,255])
    rangomin=np.array([0,0,51])
    #mascara que guarda que pixeles estan dentro del rango
    #mascara es una imagen que tiene color blanco y negro, 
    #con blanco los pixeles que estan dentro del rango y el resto negro
    mascara=cv2.inRange(frame,rangomin,rangomax)
    #opening elimina el ruido
    opening=cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    #coordenadas de bordes del objeto detectado
    x,y,w,h=cv2.boundingRect(opening)
    #dibujar rectangulo sobre la imagen
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.circle(frame,(x+w/2,y+h/2),5,(0,0,255),-1)
    #mostrar imagen
    cv2.imshow('camara',frame)
    k=cv2.waitKey(1) & 0xFF
    if k==27:
		break

#con esto podemos cerrar
cam.release()
cv2.destroyAllWindows()
