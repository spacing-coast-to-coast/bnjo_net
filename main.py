import pandas as pd
from light_tools import bands_to_wavelengths, pull_spectral_images
from sklearn.model_selection import train_test_split
from generator import SpectralImageDataGenerator
from net_def import create_3d_autoencoder
import numpy as np
from keras.models import load_model
import sys

batch_size = 32


# Load the band information
bands = pd.read_csv("enmap_bands.csv")
band_info_df = bands_to_wavelengths(bands)
wavelengths = bands['CW (nm)'].values


# Load the spectral images
spectral_images = pull_spectral_images("hyspecnet-11k/patches")


# Split the data into training and validation sets
image_paths_train, image_paths_val = train_test_split(spectral_images, test_size=0.2, random_state=42)
train_gen = SpectralImageDataGenerator(image_paths_train, batch_size=batch_size)
val_gen = SpectralImageDataGenerator(image_paths_val, batch_size=batch_size)


# Define the model
input_shape = (128, 128, 224, 1)  
autoencoder = create_3d_autoencoder(input_shape)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()


# Train the model
autoencoder.fit(
    train_gen,  # The generator provides both x and y
    epochs=1,
    batch_size=batch_size,
    shuffle=False, 
    validation_data=val_gen, 
    verbose=2
)


# Save the model
autoencoder.save("3d_autoencoder.h5")

# Load the model
loaded_model = load_model("3d_autoencoder.h5")

# Make predictions
sample_image = spectral_images[0]
sample_image = sample_image[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
reconstructed_image = loaded_model.predict(sample_image)

# Plot the original and reconstructed images
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(sample_image[0, :, :, 0, 0], cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(reconstructed_image[0, :, :, 0, 0], cmap='gray')


axes[1].set_title('Reconstructed Image')
plt.show()