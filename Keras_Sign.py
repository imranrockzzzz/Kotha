
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import cv2,os
batch_size = 128
nb_classes = 10
nb_epoch = 15

# input image dimensions
img_rows, img_cols = 80,90
oneD=7200
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
kernel_size = (5, 5)

data_augmentation=False;


#Dataset Preparation

X_train, y_train = [], []
X_test, y_test = [], []
'''Prepare the images '''


path='C:\\kotha-BanglaSignLanguageRecognition\\'

for filename in os.listdir(path):
    img=cv2.imread(path+'/'+filename)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img2=cv2.resize(img,(img_rows, img_cols))
    X_train.append(img2)
    y_train.append(int(filename[0]))


'''Divide the dataset into training and testing sample'''
x_train=X_train[:5052]
x_test=X_train[5052:len(X_train)]
y_tra=y_train[:5052]
y_test=y_train[5052:len(y_train)]
y_train=y_tra



'''Convert the created dataset into numpy array'''
X_train=np.asarray(x_train)
y_train=np.asarray(y_train)

X_test=np.asarray(x_test)
y_test=np.asarray(y_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
from matplotlib import pyplot as plt
plt.imshow(X_train[0],cmap='gray')
plt.show()
X_train = X_train.reshape(X_train.shape[0], oneD)
X_test = X_test.reshape(X_test.shape[0], oneD)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(oneD,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])
if not data_augmentation:
    history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=20,
                    verbose=1,
                    validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
else:
    print("Using data augmentation")
    datagen=ImageDataGenerator(rotation_range=90) 
    datagen.fit(X_train)
    model.fit_generator(datagen.flow(X_train,y_train,batch_size=32),
                        steps_per_epoch=batch_size,epochs=20
                        , validation_data=(X_test, y_test))
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

            
            
            
            
        