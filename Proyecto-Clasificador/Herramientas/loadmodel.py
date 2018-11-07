import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D, AveragePooling2D
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
import numpy
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import cv2

modelo='./modelo/modelo-32b-20e-2000.h5'
pesos='./modelo/pesos-32b-20e-2000.h5'
cnn=load_model(modelo)
cnn.load_weights(pesos)


cnn.summary()

data_entrenamiento='Datos2/entrenamiento'
data_validacion='Datos2/validacion'

##Parametros
#epocas=20
epocas=20
altura, longitud = 100, 100
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
	horizontal_flip=True,
	vertical_flip=True
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

#cnn.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=lr),metrics=['accuracy'])
cnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history=cnn.fit_generator(imagen_entrenamiento,steps_per_epoch=1, epochs=epocas, validation_data=imagen_validacion, validation_steps=1)

class_dictionary = imagen_entrenamiento.class_indices
print('indices= ',class_dictionary)

dir='./modelotest/'

if not os.path.exists(dir):
	os.mkdir(dir)
cnn.save('./modelotest/modelo-32b-25e-2000-2.h5')
cnn.save_weights('./modelotest/pesos-32b-25e-2000-2.h5')
