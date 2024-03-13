import pandas as pd
from utils.light_tools import bands_to_wavelengths
from utils.utils import pull_spectral_images, save_model, get_model
from sklearn.model_selection import train_test_split
from net.generator import SpectralImageDataGenerator
from net.net_def import create_3d_autoencoder
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import os


# Tensorflow configuration
# Logging for debugging
logdir = os.path.join("logs", "tf_debug_logs")
os.makedirs(logdir, exist_ok=True)
tf.debugging.experimental.enable_dump_debug_info(
    logdir,
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1
)

file_type = 'npy'
# Note: Batch size should be uniform for data generation and model training?
batch_size = 32
epochs = 10

# Note: Should be incorporated into the generator? Should be used somehow.
bands = pd.read_csv("ref_data/enmap_bands.csv")
band_info_df = bands_to_wavelengths(bands)
wavelengths = bands['CW (nm)'].values


# Load the spectral images
spectral_images, input_shape = pull_spectral_images("hyspecnet-11k/patches", type=file_type)
print(f"Loaded {len(spectral_images)} spectral images with input shape {input_shape}.")


# Split the data into training and validation sets
image_paths_train, image_paths_val = train_test_split(spectral_images, test_size=0.2, random_state=42)
train_gen = SpectralImageDataGenerator(image_paths_train, batch_size=batch_size)
val_gen = SpectralImageDataGenerator(image_paths_val, batch_size=batch_size)
print(f"Inputs set to {input_shape} with {len(image_paths_train)} training images and {len(image_paths_val)} validation images.")


# Define the model
# input_shape = (128, 128, 224, 1)  
autoencoder = create_3d_autoencoder(input_shape, )
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()


# Tensorboard callback
# tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)


# Train the model
autoencoder.fit(
    train_gen,  # The generator provides both x and y
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True, 
    validation_data=val_gen, 
    verbose='auto',
    # callbacks=[tensorboard_callback]
)


# Save the model
save_model(autoencoder, "3d_autoencoder")