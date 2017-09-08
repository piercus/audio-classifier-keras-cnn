#!/usr/bin/env python

import sys
import os
import numpy as np
from keras.models import model_from_json

def get_sample_dimensions(path):
    files = os.listdir(path)
    infilename = files[1]
    audio_path = path + '/' + infilename
    melgram = np.load(audio_path)
    return melgram.shape

def build_datasets(path, preproc = True):
    files = os.listdir(path)
    n_files = len(files)

    # pre-allocate memory for speed (old method used np.concatenate, slow)
    mel_dims = get_sample_dimensions(path=path)  # Find out the 'shape' of each data file
    X = np.zeros((n_files, mel_dims[1], mel_dims[2], mel_dims[3]))
    count = 0
    printevery = 20
    for idx2, infilename in enumerate(files):
        #print(count, infilename)
        audio_path = path + '/' + infilename
        #if (0 == idx2 % printevery):
            #print('\r file ',idx2+1," of ",n_files,": ",audio_path)
        #start = timer()
        if (preproc):
          melgram = np.load(audio_path)
          sr = 44100
        else:
          aud, sr = librosa.load(audio_path, mono=mono,sr=None)
          melgram = librosa.logamplitude(librosa.feature.melspectrogram(aud, sr=sr, n_mels=96),ref_power=1.0)[np.newaxis,np.newaxis,:,:]

        melgram = melgram[:,:,:,0:mel_dims[3]]   # just in case files are differnt sizes: clip to first file size
        X[count,:,:] = melgram
        count += 1

    return X, files

def load_model(model, weights):
    json_file = open(model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights)
    #print("Loaded model from disk")
    return loaded_model

def predict(inputpath, model_filename, weights_filename):
    model = load_model(model_filename, weights_filename)
    X, files = build_datasets(inputpath)
    Y = model.predict(X)
    # 'no_voice', 'more_mixed', 'two_mixed', 'one', 'two', 'one_mixed', 'more'
    values = (0, 0.1, 0.2, 1, 0.7, 0.3, 0.5)
    for idx2, infilename in enumerate(files):
        value = np.dot(Y[idx2], np.transpose(values))
        #print(infilename, value)
        if(value > 0.1):
            print(infilename, "contains voice", value)

if __name__ == '__main__':
    predict(sys.argv[1], sys.argv[2], sys.argv[3])
