import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def spec_AE(input_shape, learning_rate, dropout_rate=0.3, l2_penalty=0.001):
    input_layer = Input(input_shape, 1)

    # Encoder
    x = Conv1D(16, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_penalty))(input_layer)
    x = MaxPooling1D(2, padding='same')(x)
    x = Dropout(dropout_rate)(x)  # Dropout added
    x = Conv1D(8, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_penalty))(x)
    latent = MaxPooling1D(2, padding='same')(x)

    # Decoder
    x = Conv1D(8, 3, activation='relu', padding='same')(latent)
    x = UpSampling1D(2)(x)
    x = Dropout(dropout_rate)(x)  # Dropout added
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, 3, activation='linear', padding='same')(x)

    autoencoder = Model(input_layer, decoded)
    
    optimizer = Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
    
    autoencoder.summary()
    
    return autoencoder, latent

# Learning Rate Scheduler
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)