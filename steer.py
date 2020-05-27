import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import pi
import cv2
import scipy.misc
import tensorflow as tf

DATA_FOLDER = "./driving_dataset/"
DATA_FILE = os.path.join(DATA_FOLDER, "data.txt")

x = []
y = []

train_batch_pointer = 0
test_batch_pointer = 0

with open(DATA_FILE) as f:
    for line in f:
        image_name, angle = line.split()
        
        image_path = os.path.join(DATA_FOLDER, image_name)
        x.append(image_path)
        
        angle_radians = float(angle) * (pi / 180)  #converting angle into radians
        y.append(angle_radians)
y = np.array(y)
print(str(len(x))+" "+str(len(y)))

print(x[2])

split_ratio = int(len(x) * 0.8)

train_x = x[:split_ratio]
train_y = y[:split_ratio]

test_x = x[split_ratio:]
test_y = y[split_ratio:]

print(len(train_x), len(train_y), len(test_x), len(test_y))

im = cv2.imread(train_x[3000])
print(im.shape)

from keras.applications import vgg16
from keras.utils.vis_utils import plot_model

vgg16_model = vgg16.VGG16(include_top=False, weights='imagenet',input_shape=(156,455,3))

# vgg_im = vgg16_model.predict(im.reshape(1,im.shape[0],im.shape[1],3))

# square = 8
# ix = 1
# for _ in range(square):
# 	for _ in range(square):
# 		# specify subplot and turn of axis
# 		ax = plt.subplot(square, square, ix)
# 		ax.set_xticks([])
# 		ax.set_yticks([])
# 		# plot filter channel in grayscale
# 		plt.imshow(vgg_im[0, :, :, ix-1], cmap='gray')
# 		ix += 1
# # show the figure
# plt.show()

x = []
y = []
for i in range(10000):
  if(i%1000==0):
    print(i)
  im = cv2.imread(train_x[i])
  im = im[100:,:,:]/255
  vgg_im = vgg16_model.predict(im.reshape(1,156,im.shape[1],3))
  x.append(vgg_im)
  y.append(train_y[i])

print(len(x),len(y))

x1 = np.array(x)
y1 = np.array(y)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K

np.max(x1)

model = Sequential()
model.add(Flatten(input_shape=(4,14,512)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(100, activation='linear'))
model.add(Dropout(.2))
model.add(Dense(50, activation='linear'))
model.add(Dropout(.1))
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')

x1.shape

x1 = x1.reshape(x1.shape[0],4,14,512)

np.max(x1)

history = model.fit(x1/11, y1,batch_size=32,epochs=10, validation_split = 0.1, verbose = 1)

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(val_loss_values) + 1)
plt.plot(epochs, history.history['loss'], 'r', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Test loss')
plt.title('Training and test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')
plt.show()

k=-400
model.predict(x1[k].reshape(1,4,14,512)/11)

round(y1[k],2)

im = cv2.imread(train_x[k])
plt.imshow(im)
plt.grid('off')
plt.title('Predicted angle: {}, actual angle:{}'.format(str(round(model.predict(x1[k].reshape(1,4,14,512)/11)[0][0],2)), str(round(y1[k],2))))