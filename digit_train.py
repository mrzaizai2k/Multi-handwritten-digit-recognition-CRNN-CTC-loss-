
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import cv2
import os
import subprocess
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from pathlib import Path
from collections import Counter

##############Preprocessing and preparing the images for training
# Path to the data directory
data_dir = Path("data/Multi_digit_data/multi_digit_images_100/")

# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
# image chứa đường dẫn của image
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
characters = set(char for label in labels for char in label)

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)

# Batch size for training and validation
batch_size = 16

# Desired image dimensions
img_width = 128
img_height = 32

# Factor by which the image is going to be downsampled
# by the convolutional blocks. We will be using two
# convolution blocks and each block will have
# a pooling layer which downsample the features by a factor of 2.
# Hence total downsampling factor would be 4.
downsample_factor = 4

# Maximum length of any captcha in the dataset
max_length = max([len(label) for label in labels])

#The images are loaded as grayscale and reshaped to width 256 and height 64.
#The width and height are cropped if they are greater than 256 and 64 respectively. 
#If they are smaller, then the image is padded with white pixels. 
#Finally the image is rotated clockwise to bring the image shape to (x, y).
#The image is then normalized to range [0, 1]
from helper import preprocess
# seperate train/valid data
train_size = int(0.8 * len(labels))
valid_size= int(len(labels) - train_size)

train_x = []
valid_x = []
i=0
for image in images:
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image, (128,32))
    image = image/255.
    if i < 0.8*len(images):
        train_x.append(image)
    else:
        valid_x.append(image)
    i = i+1


train_x = np.array(train_x).reshape(-1, 128, 32, 1)
valid_x = np.array(valid_x).reshape(-1, 128, 32, 1)
label_train = labels[0:train_size]
label_valid = labels[train_size:len(labels)]
#print ('\n label_train',label_train)
#print('\n label_valid',label_valid)

#plt.figure(num='multi digit',figsize=(9,18))
#for i in range(3):
#    plt.subplot(3,3,i+1) 
#    plt.title(label_valid[i])
#    plt.imshow(np.squeeze(valid_x[i,:,:,]))
#plt.show()


print ('\n train_x.shape',train_x.shape)
print ('\n valid_x.shape',valid_x.shape)

alphabets = u"0123456789' "
max_str_len = 10 # max length of input labels
num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
num_of_timestamps = 32 # max length of predicted labels


def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
        
    return np.array(label_num)

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret


#train_y contains the true labels converted to numbers and padded with -1. 
#The length of each label is equal to max_str_len.
#train_label_len contains the length of each true label (without padding)
#train_input_len contains the length of each predicted label. 
#The length of all the predicted labels is constant i.e number of timestamps - 2.
#train_output is a dummy output for ctc loss.

train_y = np.ones([train_size, max_str_len]) * -1
train_label_len = np.zeros([train_size, 1])
train_input_len = np.ones([train_size, 1]) * (num_of_timestamps-2)
train_output = np.zeros([train_size])

for i in range(train_size):
    train_label_len[i] = len(label_train[i])
    train_y[i, 0:len(label_train[i])]= label_to_num(label_train[i])  
    

valid_y = np.ones([valid_size, max_str_len]) * -1
valid_label_len = np.zeros([valid_size, 1])
valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps-2)
valid_output = np.zeros([valid_size])

for i in range(valid_size):
    valid_label_len[i] = len(label_valid[i])
    valid_y[i, 0:len(label_valid[i])]= label_to_num(label_valid[i])    

#print('\n True label_train  : ',label_train[10] , '\ntrain_y : ',train_y[10],'\ntrain_label_len : ',train_label_len[10], '\ntrain_input_len : ', train_input_len[10])
#print('\n True label_valid : ',label_valid[10] , '\ntrain_y : ',valid_y[10],'\ntrain_label_len : ',valid_label_len[10], '\ntrain_input_len : ', valid_input_len[10])
### train.loc[100, 'IDENTITY'] = labels[i]

from digitreg_model import build_model


model, model_final = build_model()
#model.summary()
#model_final.summary()

opt = keras.optimizers.Adam()
early_stopping_patience = 10

# Add early stopping

my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='model/multi_digit_model/model_{epoch:02d}.h5', 
                                       save_freq='epoch', 
                                       save_best_only=True,
                                       period = 10),
    tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    )
]

model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, 
                    optimizer=keras.optimizers.Adam(lr = 0.0001),metrics=['accuracy'],                  
                    )

history = model_final.fit(x=[train_x, train_y, train_input_len, train_label_len], y=train_output, 
                validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
                epochs=3, 
                batch_size=20,
                callbacks = my_callbacks,
                )

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()

############ test #######

preds = model.predict(valid_x)
#print('\n preds',preds)
decoded = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], 
                                   greedy=True)[0][0])
#print ('\n decoded',decoded)

print ('\n predict',num_to_label(decoded[0]))

cv2.waitKey(0)