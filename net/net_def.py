from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Cropping3D
from tensorflow.keras.models import Model

def create_3d_autoencoder(input_shape):
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

    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)

    # Autoencoder Model
    autoencoder = Model(input_img, decoded)
    return autoencoder
