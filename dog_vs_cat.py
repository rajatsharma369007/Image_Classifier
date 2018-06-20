# importing libraries
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers

# dimensions of our image and including directory of training and test set
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

datagen = ImageDataGenerator(rescale=1./255)

# automagically retrieve images and their classes for train and validation sets
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')
		
# creating convolution net
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# droping the noise
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# defining the loss function
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
			  
# generating the training
model.fit_generator(
        train_generator,
        steps_per_epoch=128,
        epochs=30,
		verbose=1,
        validation_data=validation_generator,
        validation_steps=26)
		
# to save our trained model
model.save_weights('models/basic_cnn_20_epochs.h5')

# to load our trained model
#model.load_weights('models_trained/basic_cnn_20_epochs.h5')

# to validate our trained model
model.evaluate_generator(validation_generator, 26)

while(1):
	print("enter the image directory or enter 'q' to exit :")
	dir = input()
	if(dir == "q"):	
		break
	elif((dir[-1] != 'g') and (dir[-2] != 'p') and (dir[-3] != 'j')):
		print("enter jpg file format")
		continue
	else:
		from keras.preprocessing import image
		img = image.load_img(dir,target_size=(img_width,img_height))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		images = np.vstack([x])
		classes = model.predict_classes(images,batch_size=10)
		  
		if classes == [[0]]:
			k = 'cat'
		else:
			k = 'dog'
		print(k)
