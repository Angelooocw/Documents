from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32,(3,3), input_shape=(64,64,3),activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation = 'relu'))

classifier.add(Dense(units=1, activation = 'sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('CNN_Data2/training_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
test_set = test_datagen.flow_from_directory('CNN_Data2/test_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

classifier.fit_generator(training_set,
steps_per_epoch = 10,
epochs = 2,
validation_data = test_set,
validation_steps = 20)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('img.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

class_dictionary = training_set.class_indices

print('indices=',class_dictionary)

if result[0][0] == 1:
	prediction = 'dog'
	print 'dog'
else:
	prediction = 'cat'
	print 'cat'