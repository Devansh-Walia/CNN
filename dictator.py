# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:54:58 2020

@author: Devansh Walia
"""


#let's do this right
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#classifier obj creation
runner = Sequential()
# now we add the magic
runner.add(Conv2D(64,3,3, input_shape=(128, 128,3), activation='relu'))
runner.add(MaxPooling2D(pool_size=(2,2)))
runner.add(Conv2D(64,3,3, activation='relu'))
runner.add(MaxPooling2D(pool_size=(2,2)))
runner.add(Conv2D(64,3,3, activation='relu'))
runner.add(MaxPooling2D(pool_size=(2,2)))
runner.add(Flatten())

#connect this to classic ann model
runner.add(Dense(64,activation ='relu'))
runner.add(Dense(128,activation ='relu'))
runner.add(Dense(256,activation ='relu'))
runner.add(Dense(1,activation ='sigmoid'))

#giving life to our the bad boy
runner.compile(optimizer = 'adam', loss = 'binary_crossentropy',
               metrics = ['accuracy'])

#from keras.io to help simplify the code for image manipulation
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

runner.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 1,
                         validation_data = test_set,
                         nb_val_samples = 2000)
