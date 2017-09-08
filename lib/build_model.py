from keras.models import Sequential
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU

def build_model(X,Y,nb_classes):
    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size
    nb_layers = 4
    input_shape = (1, X.shape[2], X.shape[3])

    model = Sequential()
    print(X.shape[2], X.shape[3])
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='valid', input_shape=input_shape, data_format="channels_first"))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    for layer in range(nb_layers-1):
        model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))
        model.add(BatchNormalization(axis=1))
        model.add(ELU(alpha=1.0))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model
