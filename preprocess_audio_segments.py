#!/usr/bin/env python

from __future__ import print_function

'''
Voice activity detection
'''

import numpy as np
import librosa
import librosa.display
import os
import sys

def preprocess_dataset(inpath, outpath):

    if not os.path.exists(outpath):
        os.mkdir( outpath, 0755 );   # make a new directory for preproc'd files

    files = os.listdir(inpath)
    n_files = len(files)
    n_load = n_files
    printevery = 20
    for idx2, infilename in enumerate(files):
        audio_path = inpath + '/' + infilename
        if (0 == idx2 % printevery):
            print('\r file ',idx2+1," of ",n_load,": ",audio_path,sep="")
        #start = timer()
        aud, sr = librosa.load(audio_path, sr=None)
        melgram1 = librosa.feature.melspectrogram(aud, sr=sr, n_mels=96, S=np.ndarray(shape=(96,16)))
        melgram = librosa.logamplitude(melgram1,ref_power=1.0)[np.newaxis,np.newaxis,:,:]
        outfile = outpath + '/' + infilename+'.npy'
        #print(melgram.shape, melgram.shape)
        np.save(outfile,melgram)

if __name__ == '__main__':
    preprocess_dataset(sys.argv[1], sys.argv[2])
