from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense, Reshape, Conv3DTranspose, MaxPooling3D, UpSampling3D, ZeroPadding3D, Cropping3D
from kerastuner import HyperModel
import numpy as np

class AutoencoderHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        print("Building model with hyperparameters:")
        inputs = Input(shape=self.input_shape)
        x = inputs
        
        # Encoder
        for i in range(hp.Int('conv_layers', 1, 3)): # This means will test models with 1, 2, or 3 Conv3D layers
            x = Conv3D(filters=hp.Int('filters_' + str(i), 16, 64, step=16), 
                       kernel_size=(3, 3, 3), padding='same', activation='relu')(x) # Filters ranging from 16-64 with step size 16. (e.g. 16, 32, 48, 64)
            x = MaxPooling3D(pool_size=(2, 2, 2), padding='valid')(x)
        
        volume_size = x.shape[1:]
        x = Flatten()(x)
        
        # Vector Transition Dense layers
        x = Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu')(x) # Layer size can range from 64-256 with step size 64. (e.g. 64, 128, 192, 256)
        
        # Decoder
        x = Dense(np.prod(volume_size), activation='relu')(x)
        x = Reshape(target_shape=volume_size)(x)
        
        for i in range(hp.Int('convT_layers', 1, 3)):
            x = Conv3DTranspose(filters=hp.Int('filtersT_' + str(i), 16, 64, step=16), 
                                kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
            x = UpSampling3D(size=(2, 2, 2))(x)

        decoded = Conv3DTranspose(filters=1, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
        
        model = Model(inputs, decoded)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        return model

def v0_3d_autoencoder(input_shape):
    # Encoder
    input_img = Input(shape=input_shape)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling3D((2, 2, 2), padding='same')(x)
    
    # Decoder
    x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((2, 2, 2))(x)
    
    # Conditional application of Cropping3D
    if input_shape == (128, 128, 202, 1):
        print("Using 202 bands")
        cropping_layer = Cropping3D(cropping=((0, 0), (0, 0), (1, 1)))
        x = cropping_layer(x)
    elif input_shape == (128, 128, 224, 1):
        print("Using 224 bands")     

    decoded = Conv3D(1, (3, 3, 3), activation='relu', padding='same')(x)

    # Autoencoder Model
    autoencoder = Model(input_img, decoded)
    return autoencoder

def v1_3d_autoencoder(input_shape):
    # Encoder
    inputs = Input(shape=input_shape)
    x = Conv3D(filters=21, kernel_size=(3, 3, 3), padding='same', activation='relu')(inputs)
    x = Conv3D(filters=11, kernel_size=(2, 2, 2), padding='same', activation='relu')(x)
    x = Conv3D(filters=7, kernel_size=(1, 1, 1), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='valid')(x)
    volume_size = x.shape[1:] # For reshaping later
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(8, activation='linear')(x)  # Linear activation on the feature layer
    encoded = Dense(2, activation='relu')(x)

    # Decoder
    x = Dense(np.prod(volume_size), activation='relu')(encoded)  # Reshaping to the flattened input size
    x = Reshape(target_shape=volume_size)(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = ZeroPadding3D(((1,0), (1,0), (1,0)))(x)
    x = Cropping3D(cropping=(((1,0), (1,0), (1,0))))(x)
    x = Conv3DTranspose(filters=7, kernel_size=(1, 1, 1), padding='same', activation='relu')(x)
    x = Conv3DTranspose(filters=11, kernel_size=(2, 2, 2), padding='same', activation='relu')(x) 
    decoded = Conv3DTranspose(filters=1, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)

    # Define the autoencoder model
    autoencoder = Model(inputs, decoded)
    return autoencoder


