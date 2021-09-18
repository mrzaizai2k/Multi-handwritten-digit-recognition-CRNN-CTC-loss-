import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from helper import preprocess

from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#y_true: tensor (samples, max_string_length) containing the truth labels.
#y_pred: tensor (samples, time_steps, num_categories) containing the prediction, or output of the softmax.
#input_length: tensor (samples, 1) containing the sequence length of slices coming out from RNN for each batch item in y_pred.
#label_length: tensor (samples, 1) containing the sequence length of label for each batch item in y_true.


def build_model(img_width = 128,img_height = 32, max_str_len = 10):
    # Inputs to the model

    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    y_pred = layers.Dense(10 + 1, activation="softmax", name="dense2")(x) # y pred
    model = keras.models.Model(inputs=input_img, outputs=y_pred, name="functional_1")

    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage
        y_pred = y_pred[:, 2:, :]
        return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

    labels = layers.Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')

    ctc_loss = keras.layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model_final = keras.models.Model(inputs=[input_img, labels, input_length, label_length], outputs=ctc_loss, name = "ocr_model_v1")
    
    return model, model_final

######### Test model#######
def main():
    model, modelfinal = build_model()
    model.load_weights('model\mymodel.h5')
    model.summary()
    for filename in os.listdir('doc/MSSV_input'):
        print ('\n................................\n')
        print ('filename',filename)
        filepath  = os.path.join("doc/MSSV_input"+ '/', filename)
    
        image = cv2.imread(filepath, cv2.COLOR_BGR2GRAY)
        print ('image shape',image.shape)

        image = preprocess(image,(128,32))
        image = image/255.
        pred = model.predict(image.reshape(1, 128, 32, 1))
        decoded = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                            greedy=True)[0][0])
        digit_recognized = ( "".join( str(e) for e in decoded[0] ) )

        print('\nMSSV_Recognized:\n'+ digit_recognized)

if __name__ == '__main__':
    main()
    

