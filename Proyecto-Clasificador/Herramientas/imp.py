import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D, AveragePooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import applications
import matplotlib.pyplot as plt
import numpy

######
#implementar en el pc de la U, en el note explota
######
def modelo():
	vgg = applications.vgg16.VGG16()

	cnn=Sequential()
	for capa in vgg.layers:
		cnn.add(capa)

	cnn.pop()
	for layer in cnn.layers:
		layer.trainable=False

	cnn.add(Dense(11, activation='softmax'))

	return cnn

K.clear_session()

data_entrenamiento='Dataset/entrenamiento'
data_validacion='Dataset/validacion'

##Parametros
#epocas=20
epocas=20
altura, longitud = 224, 224
input_shape=(100,100,3)
batch_size=32
#pasos=1000
pasos=1000
#pasos_validacion=200
pasos_validacion=200
filtrosConv1=32
filtrosConv2=64
filtrosConv3=128
filtrosConv4=256
filtrosConv5=512
tamano_filtro1=(3,3)
tamano_filtro2=(2,2)
tamano_pool=(2,2)
clases=11
lr=0.0005

##Pre procesamiento de imagenes
entrenamiento_datagen=ImageDataGenerator(
	rescale=1./255,
	shear_range=0.3,
	zoom_range=0.3,
	horizontal_flip=True
)

validacion_datagen=ImageDataGenerator(
	rescale=1./255
)

imagen_entrenamiento= entrenamiento_datagen.flow_from_directory(
	data_entrenamiento,
	target_size=(altura,longitud),
	batch_size=batch_size,
	class_mode='categorical'
)

imagen_validacion=validacion_datagen.flow_from_directory(
	data_validacion,
	target_size=(altura,longitud),
	batch_size=batch_size,
	class_mode='categorical'
)

cnn=modelo()

cnn.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

history=cnn.fit_generator(imagen_entrenamiento,steps_per_epoch=None, epochs=epocas, validation_data=imagen_validacion, validation_steps=None)

## Graficar accuracy y loss 
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

class_dictionary = imagen_entrenamiento.class_indices
print('indices= ',class_dictionary)


dir='./modelo/'

if not os.path.exists(dir):
	os.mkdir(dir)
cnn.save('./modelo/modelo-transfer-vgg.h5')
cnn.save_weights('./modelo/pesos-transfer-vgg.h5')
