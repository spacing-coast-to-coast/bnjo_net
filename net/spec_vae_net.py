import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dropout, Dense, Lambda, Flatten, Reshape, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

class VAELoss(Layer):
    def call(self, inputs):
        x, x_decoded, z_mean, z_log_var = inputs
        # Flatten x and x_decoded for loss computation to align the dimensions
        x_flattened = tf.reshape(x, [tf.shape(x)[0], -1])
        x_decoded_flattened = tf.reshape(x_decoded, [tf.shape(x_decoded)[0], -1])

        # Compute the mean squared error loss
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(x_flattened, x_decoded_flattened))
        
        # Compute the KL divergence loss
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        
        # Total VAE loss
        total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        
        self.add_loss(total_loss)
        return x_decoded

def spec_VAE(input_shape, learning_rate, dropout_rate=0.3, l2_penalty=0.001):
    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv1D(16, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_penalty))(input_layer)
    x = MaxPooling1D(2, padding='same')(x)
    x = Dropout(dropout_rate)(x)
    x = Conv1D(8, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_penalty))(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Flatten()(x)
    z_mean = Dense(8, name='z_mean')(x)
    z_log_var = Dense(8, name='z_log_var')(x)
    z = Lambda(lambda p: p[0] + tf.exp(p[1] / 2) * tf.random.normal(shape=(tf.shape(p[0])[0], 8)))([z_mean, z_log_var])

    # Decoder
    decoder_shape = (input_shape[0] // 4, 8)
    x = Dense(decoder_shape[0] * decoder_shape[1], activation='relu')(z)
    x = Reshape(decoder_shape)(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(8, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)

    # Integrate VAELoss layer
    vae_outputs = VAELoss()([input_layer, decoded, z_mean, z_log_var])
    vae = Model(inputs=input_layer, outputs=vae_outputs)

    vae.compile(optimizer=Adam(learning_rate), loss=None)  # Loss is already added via add_loss

    vae.summary()

    return vae
