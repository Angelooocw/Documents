import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
longitud, altura = 100,100
modelo='./modelo/modelo.h5'
pesos='./modelo/pesos.h5'
cnn=load_model(modelo)
cnn.load_weights(pesos)

def prediccion(file):
	x=load_img(file,target_size=(longitud,altura))
	x=img_to_array(x)
	x=np.expand_dims(x, axis=0)

	print(x)
	arreglo=cnn.predict(x) ##[[1,0,0]]
	print arreglo
	resultado=arreglo[0]
	respuesta=np.argmax(resultado)


	print resultado

	if respuesta==0:
		print 'Alicate'
	elif respuesta==1:
		print 'Destornillador'
	elif respuesta==2:
		print 'Llave'
	elif respuesta==3:
		print 'Martillo'
	elif respuesta==4:
		print 'Taladro'
	return respuesta

prediccion('crop.jpg')

