from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Conv3DTranspose, Flatten, Reshape, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


# def hyspec_hsi_VAE(input_shape):
#     input_layer = Input(shape=input_shape)

#     # Encoder
#     x = Conv3D(filters=21, kernel_size=(3, 3, 3), padding='same', activation='relu')(input_layer)
#     x = Conv3D(filters=11, kernel_size=(2, 2, 2), padding='same', activation='relu')(x)
#     x = Conv3D(filters=7, kernel_size=(1, 1, 1), padding='same', activation='relu')(x)
#     x = MaxPooling3D(pool_size=(2, 2, 2), padding='valid')(x)
#     x = Flatten()(x)
#     x = Dense(128, activation='relu')(x)
#     z_mean = Dense(8, name='z_mean')(x)
#     z_log_var = Dense(8, name='z_log_var')(x)
#     z = Lambda(sampling, output_shape=(8,), name='z')([z_mean, z_log_var])

#     # Decoder
#     flattened_size = np.prod([input_shape[0]//2, input_shape[1]//2, input_shape[2]//2, 7])
#     x = Dense(flattened_size, activation='relu')(z)
#     x = Reshape((input_shape[0]//2, input_shape[1]//2, input_shape[2]//2, 7))(x)
#     x = Conv3DTranspose(filters=7, kernel_size=(1, 1, 1), padding='same', activation='relu')(x)
#     x = UpSampling3D(size=(2, 2, 2))(x)
#     x = Conv3DTranspose(filters=11, kernel_size=(2, 2, 2), padding='same', activation='relu')(x)
#     x = Conv3DTranspose(filters=21, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
#     decoded = Conv3DTranspose(filters=1, kernel_size=(3, 3, 3), padding='same', activation='sigmoid')(x)

#     vae = Model(input_layer, decoded, name='hyspec_hsi_vae')
#     optimizer = Adam(1e-4)
#     vae.compile(optimizer=optimizer, loss='mean_squared_error')
#     vae.summary()

#     return vae

# def sampling(args):
#     z_mean, z_log_var = args
#     batch = tf.shape(z_mean)[0]
#     dim = tf.shape(z_mean)[1]
#     epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#     return z_mean + tf.exp(0.5 * z_log_var) * epsilon


from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Reshape, Conv3DTranspose, Lambda, UpSampling3D, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf


def hyspec_hsi_VAE(input_shape):
    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv2D(filters=21, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    x = Conv2D(filters=11, kernel_size=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters=7, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    z_mean = Dense(8, name='z_mean')(x)
    z_log_var = Dense(8, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(8,), name='z')([z_mean, z_log_var])

    # Decoder
    flattened_size = np.prod([input_shape[0]//2, input_shape[1]//2, 7])
    x = Dense(flattened_size, activation='relu')(z)
    x = Reshape((input_shape[0]//2, input_shape[1]//2, 7))(x)
    x = Conv2DTranspose(filters=7, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(filters=11, kernel_size=(2, 2), padding='same', activation='relu')(x)
    x = Conv2DTranspose(filters=21, kernel_size=(3, 3), padding='same', activation='relu')(x)
    decoded = Conv2DTranspose(filters=1, kernel_size=(3, 3), padding='same', activation='sigmoid')(x)

    vae = Model(input_layer, decoded, name='hyspec_hsi_vae')
    optimizer = Adam(1e-4)
    vae.compile(optimizer=optimizer, loss='mean_squared_error')
    vae.summary()

    return vae

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


lr_reducer = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=.0001, verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=10,min_delta=0.001, verbose=1, mode='min', restore_best_weights=True)