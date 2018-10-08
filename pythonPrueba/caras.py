import numpy as np 
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
	#se obtienen las imagenes de la camara
	#read() devuelve una tupla que es almacenada en las variables ret y frame
	ret, frame = cap.read()
	#convierte BGR a GRAY
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#detecta objetos de diferentes tamanos en la imagen
	#scaleFactor=1.3   minNeighbors= 5
	faces = face_cascade.detectMultiScale(gray,1.3,5)
	for(x,y,w,h) in faces:
		#Dibuja rectangulos sobre cada cara detectada
		cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
		#region de interes, dentro del rectangulo de la cara detectada
		roi_gray = gray[y:y+h,x:x+w]
		roi_color = frame[y:y+h,x:x+w]
		#detecta los ojos
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			#dibuja rectangulos sobre los ojos
			cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0),2)

	cv2.imshow('img',frame)
	k= cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
