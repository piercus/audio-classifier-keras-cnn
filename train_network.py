from __future__ import print_function

'''
Classify sounds using database
Author: Scott H. Hawley

This is kind of a mixture of Keun Woo Choi's code https://github.com/keunwoochoi/music-auto_tagging-keras
   and the MNIST classifier at https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

Trained using Fraunhofer IDMT's database of monophonic guitar effects,
   clips were 2 seconds long, sampled at 44100 Hz
'''

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from "./lib/build_datasets" import build_datasets
from "./lib/build_model" import build_model

from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras import backend
from keras.utils import np_utils
import os
from os.path import isfile

from timeit import default_timer as timer

mono=True

if __name__ == '__main__':
    np.random.seed(1)

    # get the data
    X_train, Y_train, paths_train, X_test, Y_test, paths_test, class_names, sr = build_datasets(preproc=True)

    # make the model
    model = build_model(X_train,Y_train, nb_classes=len(class_names))
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
    model.summary()
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # Initialize weights using checkpoint if it exists. (Checkpointing requires h5py)
    load_checkpoint = True
    checkpoint_filepath = 'weights.hdf5'
    if (load_checkpoint):
        print("Looking for previous weights...")
        if ( isfile(checkpoint_filepath) ):
            print ('Checkpoint file detected. Loading weights.')
            model.load_weights(checkpoint_filepath)
        else:
            print ('No checkpoint file detected.  Starting from scratch.')
    else:
        print('Starting from scratch (no checkpoint)')
    checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, save_best_only=True)


    # train and score the model
    batch_size = 128
    nb_epoch = 100
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test), callbacks=[checkpointer])
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
