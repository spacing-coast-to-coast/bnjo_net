import tensorflow as tf
from net.spec_net import spec_AE
from net.hsi_net import hsi_3dAE

def CCA_DCCA(hsi_input_shape, spectral_input_shape, spec_lr=0.001, hsi_lr=0.001, dropout_rate=0.3, l2_penalty=0.001):
    hsi_autoencoder, _ = hsi_3dAE(input_shape=hsi_input_shape)
    hsi_autoencoder.summary()

    spec_autoencoder, _ = spec_AE(input_shape=spectral_input_shape, learning_rate=spec_lr)
    spec_autoencoder.summary()

    class CCA_DCCA_Model(tf.keras.Model):
        def __init__(self, hsi_encoder, spectral_encoder, **kwargs):
            super(CCA_DCCA_Model, self).__init__(**kwargs)
            self.hsi_encoder = hsi_encoder
            self.spectral_encoder = spectral_encoder

        def call(self, inputs):
            hsi_input, spectral_input = inputs
            hsi_latent = self.hsi_encoder(hsi_input)
            spectral_latent = self.spectral_encoder(spectral_input)
            print("HSI Latent Shape:", hsi_latent.shape)
            print("Spectral Latent Shape:", spectral_latent.shape)
            return hsi_latent, spectral_latent

        def train_step(self, data):
            hsi_input, spectral_input = data

            with tf.GradientTape() as tape:
                # Obtain the latent representations for both inputs
                hsi_latent, spectral_latent = self((hsi_input, spectral_input), training=True)
                
                # Normalize the latent representations to have unit norm, which is a requirement for CCA
                hsi_latent_normalized = tf.math.l2_normalize(hsi_latent, axis=1)
                spectral_latent_normalized = tf.math.l2_normalize(spectral_latent, axis=1)
                
                # Note: Should explore an actual CCA/DCCA implementation here
                loss = tf.reduce_mean(tf.square(hsi_latent_normalized - spectral_latent_normalized))

            # Compute gradients and apply 
            trainable_vars = self.hsi_encoder.trainable_variables + self.spectral_encoder.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            print("Current Loss:", loss.numpy())
            return {"loss": loss}

    cca_dcca_model = CCA_DCCA_Model(hsi_autoencoder, spec_autoencoder)
    cca_dcca_model.compile(optimizer='adam')
    cca_dcca_model.summary()

    return cca_dcca_model
