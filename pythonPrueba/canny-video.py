import numpy as np 
import cv2

#cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture("imagenes/prueba.mp4")
while (True):

	ret,frame = cam.read()
	#si llega al final del video sale
	if not ret:
		break

	#imagen en escala de grises
	gray = 	cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#suavizado de imagen convertida
	blur = cv2.GaussianBlur(gray, (21,21), 0)
	#Metodo para detectar bordes mediante un umbral
	#Dos maneras de detectar bordes en video

	#ret, thresh_img = cv2.threshold(blur,91,255,cv2.THRESH_BINARY)
	#cv2.imshow("umbral",thresh_img)
	#(_,contornos,_) = cv2.findContours(thresh_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	#Deteccion de bordes con canny
	canny = cv2.Canny(blur, 50, 150)
	cv2.imshow("canny",canny)
	#deteccion de contornos
	(_,contornos,_) = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	#Dibujar contornos sobre video
	for c in contornos:
		cv2.drawContours(frame,[c], -1, (0,0,255),2)
	
	#cv2.imshow("pp",thresh_img)
	cv2.imshow('camara',frame)
 
	k=cv2.waitKey(1) & 0xFF
	if k == 27:
		break

camara.release()
cv2.destroyAllWindows()

