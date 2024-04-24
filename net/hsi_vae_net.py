import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Lambda, Conv3D, MaxPooling3D, UpSampling3D, Conv3DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np


def hsi_VAE(input_shape):
    input_layer = Input(shape=input_shape)

    # Encoderx
    x = Conv3D(filters=21, kernel_size=(3, 3, 3), padding='same', activation='relu')(input_layer)
    x = Conv3D(filters=11, kernel_size=(2, 2, 2), padding='same', activation='relu')(x)
    x = Conv3D(filters=7, kernel_size=(1, 1, 1), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    z_mean = Dense(8, name='z_mean')(x)
    z_log_var = Dense(8, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(8,), name='z')([z_mean, z_log_var])

    # Decoder
    x = Dense(np.prod([input_shape[0]//4, input_shape[1]//4, input_shape[2]//4, 16]), activation='relu')(z)
    x = Reshape((input_shape[0]//4, input_shape[1]//4, input_shape[2]//4, 16))(x)
    x = Conv3DTranspose(filters=7, kernel_size=(1, 1, 1), padding='same', activation='relu')(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3DTranspose(filters=11, kernel_size=(2, 2, 2), padding='same', activation='relu')(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    decoded = Conv3DTranspose(filters=21, kernel_size=(3, 3, 3), padding='same', activation='sigmoid')(x)

    # Instantiate VAE model
    vae = Model(input_layer, decoded, name='hsi_vae')

    # # Add VAE loss
    # decoded_with_loss = VAE_Loss()([input_layer, decoded, z_mean, z_log_var])

    # # Instantiate VAE model using the output from the VAE_Loss layer
    # vae = Model(input_layer, decoded_with_loss, name='hsi_vae')
    
    # Compile the model
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
