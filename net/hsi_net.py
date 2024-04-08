import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.layers import Reshape, UpSampling3D, ZeroPadding3D, Cropping3D, Conv3DTranspose
from tensorflow.keras.models import Model
import numpy as np


def hsi_3dAE(input_shape):
    # Encoder
    inputs = Input(shape=input_shape)

    # Encoder based on the architectuure described in the paper
    # "3D-CNN and Autoencoder-Based Gas Detection in Hyperspectral Images" by Ozdemir and Koz, 2023
    # They describe three successive convolutional layers before encoding with 21, 11, and 7 filters, respectively.
    x = Conv3D(filters=21, kernel_size=(3, 3, 3), padding='same', activation='relu')(inputs)
    x = Conv3D(filters=11, kernel_size=(2, 2, 2), padding='same', activation='relu')(x)
    x = Conv3D(filters=7, kernel_size=(1, 1, 1), padding='same', activation='relu')(x)
    # Pooling layer added for dimensionality reduction
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='valid')(x)
    volume_size = x.shape[1:] # For reshaping later
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(8, activation='linear')(x)  # Linear activation on the feature layer

    # The latent space dimension is designed to be shared amongst other encoders.
    # It is a 2D layer that is used to compare the latent spaces of different encoders.
    # The technique is described in the paper "An end-to-end deep learning framework for
    # translating mass spectra to de-novo molecules" by Litsa et al., 2023
    latent_space = Dense(2, activation='relu')(x)

    # Decoder
    x = Dense(np.prod(volume_size), activation='relu')(latent_space)  # Reshaping to the flattened input size
    x = Reshape(target_shape=volume_size)(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = ZeroPadding3D(((1,0), (1,0), (1,0)))(x)
    x = Cropping3D(cropping=(((1,0), (1,0), (1,0))))(x)
    x = Conv3DTranspose(filters=7, kernel_size=(1, 1, 1), padding='same', activation='relu')(x)
    x = Conv3DTranspose(filters=11, kernel_size=(2, 2, 2), padding='same', activation='relu')(x) 
    decoded = Conv3DTranspose(filters=1, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)

    # Define the autoencoder model
    autoencoder = Model(inputs, decoded)
    return autoencoder, latent_space
