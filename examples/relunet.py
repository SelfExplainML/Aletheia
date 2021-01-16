import numpy as np 
import tensorflow as tf
from tensorflow import keras


def MLPClassifier(input_dim, hidden_layer_sizes, regl1=0, regl2=0, lr=0.001, random_state=0):

    np.random.seed(random_state) 
    tf.random.set_seed(random_state)

    hidden_layer_sizes = [input_dim] + hidden_layer_sizes
    model = keras.models.Sequential()

    if (regl1 > 0) & (regl2 == 0):
        regularizer = keras.regularizers.l1(regl1)
    elif (regl1 == 0) & (regl2 > 0):
        regularizer = keras.regularizers.l2(regl2)
    else:
        regularizer = keras.regularizers.l1_l2(l1=regl1, l2=regl2)
    
    for i in range(len(hidden_layer_sizes) - 1):
        model.add(keras.layers.Dense(
            input_dim=hidden_layer_sizes[i],
            units=hidden_layer_sizes[i + 1],
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            activation='relu',
            kernel_regularizer=regularizer))

    model.add(keras.layers.Dense(
            units=1,
            input_dim=hidden_layer_sizes[-1],
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            activation='sigmoid',
            kernel_regularizer=regularizer))

    # compile model
    model.compile(optimizer=keras.optimizers.Adam(lr=lr),
                    loss='binary_crossentropy')
    return model


def MLPRegressor(input_dim, hidden_layer_sizes, regl1=0, regl2=0, lr=0.001, random_state=0):

    np.random.seed(random_state) 
    tf.random.set_seed(random_state)

    hidden_layer_sizes = [input_dim] + hidden_layer_sizes
    model = keras.models.Sequential()

    if (regl1 > 0) & (regl2 == 0):
        regularizer = keras.regularizers.l1(regl1)
    elif (regl1 == 0) & (regl2 > 0):
        regularizer = keras.regularizers.l2(regl2)
    else:
        regularizer = keras.regularizers.l1_l2(l1=regl1, l2=regl2)

    for i in range(len(hidden_layer_sizes) - 1):
        model.add(keras.layers.Dense(
            input_dim=hidden_layer_sizes[i],
            units=hidden_layer_sizes[i + 1],
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            activation='relu',
            kernel_regularizer=regularizer))

    model.add(keras.layers.Dense(
            units=1,
            input_dim=hidden_layer_sizes[-1],
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            activation='linear',
            kernel_regularizer=regularizer))

    # compile model
    model.compile(optimizer=keras.optimizers.Adam(lr=lr),
                    loss='mean_squared_error')
    return model
