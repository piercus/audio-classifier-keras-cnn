from get_total_files import get_total_files
from get_class_names import get_class_names
from get_sample_dimensions import get_sample_dimensions
from encode_class import encode_class
from shuffle_XY_paths import shuffle_XY_paths

import numpy as np
import os

'''
So we make the training & testing datasets here, and we do it separately.
Why not just make one big dataset, shuffle, and then split into train & test?
because we want to make sure statistics in training & testing are as similar as possible
'''
def build_datasets(train_percentage=0.8, preproc=False):
    if (preproc):
        path = "Preproc/"
    else:
        path = "Samples/"

    class_names = get_class_names(path=path)
    print("class_names = ",class_names)

    total_files, total_train, total_test = get_total_files(path=path, train_percentage=train_percentage)
    print("total files = ",total_files)

    nb_classes = len(class_names)

    # pre-allocate memory for speed (old method used np.concatenate, slow)
    mel_dims = get_sample_dimensions(path=path)  # Find out the 'shape' of each data file
    X_train = np.zeros((total_train, mel_dims[1], mel_dims[2], mel_dims[3]))
    Y_train = np.zeros((total_train, nb_classes))
    X_test = np.zeros((total_test, mel_dims[1], mel_dims[2], mel_dims[3]))
    Y_test = np.zeros((total_test, nb_classes))
    paths_train = []
    paths_test = []

    train_count = 0
    test_count = 0
    for idx, classname in enumerate(class_names):
        this_Y = np.array(encode_class(classname,class_names) )
        this_Y = this_Y[np.newaxis,:]
        class_files = os.listdir(path+classname)
        n_files = len(class_files)
        n_load =  n_files
        n_train = int(train_percentage * n_load)
        printevery = 100
        print("")
        for idx2, infilename in enumerate(class_files[0:n_load]):
            audio_path = path + classname + '/' + infilename
            if (0 == idx2 % printevery):
                print('\r Loading class: {:14s} ({:2d} of {:2d} classes)'.format(classname,idx+1,nb_classes),
                  ", file ",idx2+1," of ",n_load,": ",audio_path)
            #start = timer()
            if (preproc):
              melgram = np.load(audio_path)
              sr = 44100
            else:
              aud, sr = librosa.load(audio_path, mono=mono,sr=None)
              melgram = librosa.logamplitude(librosa.feature.melspectrogram(aud, sr=sr, n_mels=96),ref_power=1.0)[np.newaxis,np.newaxis,:,:]

            melgram = melgram[:,:,:,0:mel_dims[3]]   # just in case files are differnt sizes: clip to first file size

            #end = timer()
            #print("time = ",end - start)
            if (idx2 < n_train):
                # concatenate is SLOW for big datasets; use pre-allocated instead
                #X_train = np.concatenate((X_train, melgram), axis=0)
                #Y_train = np.concatenate((Y_train, this_Y), axis=0)
                X_train[train_count,:,:] = melgram
                Y_train[train_count,:] = this_Y
                paths_train.append(audio_path)     # list-appending is still fast. (??)
                train_count += 1
            else:
                X_test[test_count,:,:] = melgram
                Y_test[test_count,:] = this_Y
                #X_test = np.concatenate((X_test, melgram), axis=0)
                #Y_test = np.concatenate((Y_test, this_Y), axis=0)
                paths_test.append(audio_path)
                test_count += 1
        print("")

    print("Shuffling order of data...")
    X_train, Y_train, paths_train = shuffle_XY_paths(X_train, Y_train, paths_train)
    X_test, Y_test, paths_test = shuffle_XY_paths(X_test, Y_test, paths_test)

    return X_train, Y_train, paths_train, X_test, Y_test, paths_test, class_names, sr
