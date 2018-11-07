from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D, AveragePooling2D

filtrosConv1=32
filtrosConv2=64
filtrosConv3=128
filtrosConv4=256
filtrosConv5=512
tamano_filtro1=(3,3)
tamano_filtro2=(2,2)
input_shape=(100,100,3)
tamano_pool=(2,2)


cnn=Sequential()
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
cnn.add(Dense(11,activation='softmax'))

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


cnn.summary()