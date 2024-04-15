import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dropout, Flatten, Dense, Reshape, Lambda
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Conv3DTranspose
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import numpy as np



# def sampling(args):
#     """Reparameterization trick by sampling from an isotropic unit Gaussian."""
#     z_mean, z_log_var = args
#     batch = K.shape(z_mean)[0]
#     dim = K.int_shape(z_mean)[1]
#     epsilon = K.random_normal(shape=(batch, dim))
#     return z_mean + K.exp(0.5 * z_log_var) * epsilon

# class VAE_Loss(Layer):
#     def __init__(self, **kwargs):
#         super(VAE_Loss, self).__init__(**kwargs)

#     def compute_output_shape(self, input_shape):
#         # Example: return the shape of the 'decoded' input tensor
#         return input_shape[1]

#     def call(self, inputs):
#         input_layer, decoded, z_mean, z_log_var = inputs

#         print("input_layer shape:", input_layer.shape)
#         print("decoded shape:", decoded.shape)
#         print("z_mean shape:", z_mean.shape)
#         print("z_log_var shape:", z_log_var.shape)

#         # Adjust reduction axis to match the mean_squared_error output dimensions
#         reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(input_layer, decoded), axis=1)
#         kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
#         vae_loss = reconstruction_loss + kl_loss
#         self.add_loss(vae_loss)
#         return decoded

# def sampling(args):
#     z_mean, z_log_var = args
#     epsilon = tf.random.normal(shape=tf.shape(z_mean))
#     return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Compute VAE loss
def compute_vae_loss(input_layer, decoded, z_mean, z_log_var):
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(input_layer, decoded), axis=(1, 2)))
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae_loss = reconstruction_loss + kl_loss
    return vae_loss

@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        decoded, z_mean, z_log_var = model(x, training=True)
        loss = compute_vae_loss(x, decoded, z_mean, z_log_var)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def spec_VAE(input_shape, learning_rate=0.001, dropout_rate=0.4, l2_penalty=0.001):
    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv1D(16, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_penalty))(input_layer)
    x = MaxPooling1D(2, padding='same')(x)
    x = Dropout(dropout_rate)(x)
    x = Conv1D(8, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_penalty))(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Flatten()(x)
    z_mean = Dense(8, name='z_mean')(x)
    z_log_var = Dense(8, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(8,), name='z')([z_mean, z_log_var])

    # Decoder
    decoder_dense_units = (input_shape[0] // 4) * 8  # Adjusted to match the encoder's flattening point
    x = Dense(decoder_dense_units, activation='relu')(z)
    x = Reshape(((input_shape[0] // 4), 8))(x)  # Reshape to reverse the flattening in the encoder
    
    # Reverse the operations of the encoder
    x = UpSampling1D(2)(x)
    x = Conv1D(8, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    
    # Final Conv1D to match the original input channels
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)

    # Instantiate VAE model
    vae = Model(input_layer, decoded, name='spec_vae')

    # # Add VAE loss
    # decoded_with_loss = VAE_Loss()([input_layer, decoded, z_mean, z_log_var])

    # # Instantiate VAE model using the output from the VAE_Loss layer
    # vae = Model(input_layer, decoded_with_loss, name='spec_vae')

    optimizer = Adam(learning_rate=learning_rate)
    
    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            decoded = vae(x, training=True)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.mean_squared_error(x, decoded), axis=(1, 2))
            )
            kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
            loss = reconstruction_loss + kl_loss
        gradients = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        return loss

    # Return the model and the custom training step
    return vae, train_step


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


# def sampling(args):
#     z_mean, z_log_var = args
#     epsilon = tf.random.normal(shape=tf.shape(z_mean))
#     return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def sample_z(z_mean, z_log_var):
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon



class SpecVAE(tf.keras.Model):
    def __init__(self, input_shape, learning_rate=0.001, dropout_rate=0.4, l2_penalty=0.001):
        super(SpecVAE, self).__init__()
        self.input_shape = input_shape
        
        # Encoder definition
        self.encoder = tf.keras.Sequential([
            Input(shape=self.input_shape),
            Conv1D(16, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_penalty)),
            MaxPooling1D(2, padding='same'),
            Dropout(dropout_rate),
            Conv1D(8, 3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_penalty)),
            MaxPooling1D(2, padding='same'),
            Flatten(),
            Dense(16, activation='relu')
        ])
        
        # Latent space definition
        self.z_mean_layer = Dense(8, name='z_mean')
        self.z_log_var_layer = Dense(8, name='z_log_var')
        
        # Decoder definition
        target_shape = (self.input_shape[0] // 4, 8)
        decoder_dense_units = target_shape[0] * target_shape[1]
        self.decoder = tf.keras.Sequential([
            Dense(decoder_dense_units, activation='relu'),
            Reshape(target_shape),
            UpSampling1D(2),
            Conv1D(8, 3, activation='relu', padding='same'),
            UpSampling1D(2),
            Conv1D(16, 3, activation='relu', padding='same'),
            Conv1D(1, 3, activation='sigmoid', padding='same')
        ])
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def call(self, inputs):
        encoded = self.encoder(inputs)
        self.z_mean = self.z_mean_layer(encoded)
        self.z_log_var = self.z_log_var_layer(encoded)
        z = sampling([self.z_mean, self.z_log_var])
        decoded = self.decoder(z)
        return decoded

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            decoded = self.call(x)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, decoded), axis=-1)
            kl_loss = -0.5 * tf.reduce_sum(1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var), axis=-1)
            loss = reconstruction_loss + kl_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
