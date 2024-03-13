import pandas as pd
from utils.light_tools import bands_to_wavelengths
from utils.utils import setup_model, pull_spectral_images, save_model
from sklearn.model_selection import train_test_split
from net.generator import SpectralImageDataGenerator
from net.net_def import create_3d_autoencoder


# Model Configuration
model_name = '3d_autoencoder'
input_file_type = 'npy'
batch_size = 32
epochs = 10
use_callbacks = ['tf_logs', 'keras_checkpoints']


# Note: Should be incorporated into the generator? Should be used somehow.
bands = pd.read_csv("ref_data/enmap_bands.csv")
band_info_df = bands_to_wavelengths(bands)
wavelengths = bands['CW (nm)'].values


# Load the spectral images
spectral_images, input_shape = pull_spectral_images("hyspecnet-11k/patches", type=input_file_type)
print(f"Loaded {len(spectral_images)} spectral images with input shape {input_shape}.")


# Split the data into training and validation sets
image_paths_train, image_paths_val = train_test_split(spectral_images, test_size=0.2, random_state=42)
train_gen = SpectralImageDataGenerator(image_paths_train, batch_size=batch_size)
val_gen = SpectralImageDataGenerator(image_paths_val, batch_size=batch_size)
print(f"Inputs set to {input_shape} with {len(image_paths_train)} training images and {len(image_paths_val)} validation images.")


# Define the model
# input_shape = (128, 128, 224, 1)  
autoencoder = create_3d_autoencoder(input_shape)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()

# Setup the model directory and callbacks
model_dir, callbacks = setup_model(model_name, use_callbacks=use_callbacks)
print(f"Using callbacks: {callbacks}")


# Train the model
print(f"Training model {model_name} with input shape {input_shape} and {len(image_paths_train)} images.")
autoencoder.fit(
    train_gen,  # The generator provides both x and y
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True, 
    validation_data=val_gen, 
    verbose='auto',
    callbacks=callbacks
)


# Save the model
saved_model = save_model(autoencoder, model_name, model_dir)  
print(f"Model and accompanying info saved to {saved_model}")