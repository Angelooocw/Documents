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



K.clear_session()

data_entrenamiento='Dataset/entrenamiento'
data_validacion='Dataset/validacion'

##Parametros
#epocas=20
epocas=20
altura, longitud = 150, 150
input_shape=(150,150,3)
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

#Crear la red neuronal CNN
###Red Basica

"""cnn=Sequential()
cnn.add(Convolution2D(filtrosConv1,tamano_filtro1,padding='same', input_shape=(altura,longitud,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2,tamano_filtro2,padding='same',activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(256,activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases,activation='softmax'))"""

###Mini VGG

"""cnn=Sequential()
cnn.add(Convolution2D(filtrosConv2,tamano_filtro1,input_shape=input_shape,padding='same',activation='relu'))
cnn.add(Convolution2D(filtrosConv2,tamano_filtro1,activation='relu',padding='same'))
#cnn.add(MaxPooling2D(pool_size=tamano_pool, strides=(2,2)))
cnn.add(AveragePooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv3,tamano_filtro1,activation='relu',padding='same'))
cnn.add(Convolution2D(filtrosConv3,tamano_filtro1,activation='relu',padding='same'))
#cnn.add(MaxPooling2D(pool_size=tamano_pool, strides=(2,2)))
cnn.add(AveragePooling2D(pool_size=tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(512,activation='relu'))
cnn.add(Dense(256,activation='relu'))
cnn.add(Dense(clases,activation='softmax'))"""

####Prueba RED8

cnn=Sequential()
cnn.add(Convolution2D(filtrosConv1,tamano_filtro1,input_shape=input_shape,padding='same',activation='relu'))
cnn.add(Convolution2D(filtrosConv1,tamano_filtro1,activation='relu',padding='same'))
cnn.add(MaxPooling2D(pool_size=tamano_pool, strides=(2,2)))
#cnn.add(Dropout(0.5))
#cnn.add(Dropout(0.25))

cnn.add(Convolution2D(filtrosConv2,tamano_filtro1,input_shape=input_shape,padding='same',activation='relu'))
cnn.add(Convolution2D(filtrosConv2,tamano_filtro1,activation='relu',padding='same'))
cnn.add(MaxPooling2D(pool_size=tamano_pool, strides=(2,2)))
#cnn.add(Dropout(0.5))
#cnn.add(Dropout(0.25))

cnn.add(Convolution2D(filtrosConv2,tamano_filtro1,input_shape=input_shape,padding='same',activation='relu'))
cnn.add(Convolution2D(filtrosConv2,tamano_filtro1,activation='relu',padding='same'))

cnn.add(MaxPooling2D(pool_size=tamano_pool, strides=(2,2)))
#cnn.add(Dropout(0.25))

cnn.add(Flatten())
cnn.add(Dense(512,activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases,activation='softmax'))


"""cnn=Sequential()
cnn.add(Convolution2D(filtrosConv2,tamano_filtro1,input_shape=input_shape,padding='same',activation='relu'))
cnn.add(Convolution2D(filtrosConv2,tamano_filtro1,activation='relu',padding='same'))
cnn.add(MaxPooling2D(pool_size=tamano_pool, strides=(2,2)))

cnn.add(Convolution2D(filtrosConv3,tamano_filtro1,activation='relu',padding='same'))
cnn.add(Convolution2D(filtrosConv3,tamano_filtro1,activation='relu',padding='same'))
cnn.add(MaxPooling2D(pool_size=tamano_pool, strides=(2,2)))

cnn.add(Convolution2D(filtrosConv4,tamano_filtro1,activation='relu',padding='same'))
cnn.add(Convolution2D(filtrosConv4,tamano_filtro1,activation='relu',padding='same'))
cnn.add(Convolution2D(filtrosConv4,tamano_filtro1,activation='relu',padding='same'))
cnn.add(MaxPooling2D(pool_size=tamano_pool, strides=(2,2)))

cnn.add(Convolution2D(filtrosConv5,tamano_filtro1,activation='relu',padding='same'))
cnn.add(Convolution2D(filtrosConv5,tamano_filtro1,activation='relu',padding='same'))
cnn.add(Convolution2D(filtrosConv5,tamano_filtro1,activation='relu',padding='same'))
cnn.add(MaxPooling2D(pool_size=tamano_pool, strides=(2,2)))

cnn.add(Convolution2D(filtrosConv5,tamano_filtro1,activation='relu',padding='same'))
cnn.add(Convolution2D(filtrosConv5,tamano_filtro1,activation='relu',padding='same'))
cnn.add(Convolution2D(filtrosConv5,tamano_filtro1,activation='relu',padding='same'))
cnn.add(MaxPooling2D(pool_size=tamano_pool, strides=(2,2)))

cnn.add(Flatten())
cnn.add(Dense(4096,activation='relu'))
cnn.add(Dense(4096,activation='relu'))
cnn.add(Dense(clases,activation='softmax'))"""

######

cnn.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=lr),metrics=['accuracy'])
#cnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

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
cnn.save('./modelo/modelo-32b-25e-2000.h5')
cnn.save_weights('./modelo/pesos-32b-25e-2000.h5')
