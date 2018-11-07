import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from keras import models

epocas=20
altura, longitud = 100, 100
batch_size=32
#pasos=1000
pasos=1000
#pasos_validacion=200
pasos_validacion=100
filtrosConv1=32
filtrosConv2=64
filtrosConv3=128
tamano_filtro1=(3,3)
tamano_filtro2=(2,2)
tamano_pool=(2,2)
clases=11
lr=0.0005

cnn=Sequential()
cnn.add(Convolution2D(filtrosConv1,tamano_filtro1,padding='same', input_shape=(altura,longitud,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2,tamano_filtro2,padding='same',activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv3,tamano_filtro2,padding='same',activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(256,activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases,activation='softmax'))

cnn.summary()
