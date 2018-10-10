import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud, altura = 100,100
modelo='./modelo/modelo.h5'
pesos='./modelo/pesos.h5'
cnn=load_model(modelo)
cnn.load_weights(pesos)

def prediccion(file):
	x=load_img(file,target_size=(longitud,altura))
	x=img_to_array(x)
	x=np.expand_dims(x, axis=0)


	arreglo=cnn.predict(x) ##[[1,0,0]]
	print arreglo
	resultado=arreglo[0]
	respuesta=np.argmax(resultado)

#('indices= ', {'gato': 0, 'gorila': 1, 'pajaro': 2, 'perro': 3})
	
	
	if respuesta==0:
		print resultado
		print 'Gato'
	elif respuesta==1:
		print resultado
		print 'Gorila'
	elif respuesta==2:
		print resultado
		print 'Pajaro'
	elif respuesta==3:
		print resultado
		print 'Perro'
	return respuesta

prediccion('p12.jpg')
