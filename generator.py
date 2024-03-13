import numpy as np
import rasterio
from keras.utils import Sequence
from pathlib import Path

class SpectralImageDataGenerator(Sequence):
    def __init__(self, image_paths, batch_size=32, shuffle=False, **kwargs):
        """
        image_paths: List of paths to the spectral images.
        batch_size: Number of images to process in a batch.
        shuffle: Whether to shuffle the order of images at the end of each epoch.
        """
        print(f"Creating generator with {len(image_paths)} images. Batch size: {batch_size} Shuffle: {shuffle}")
        super().__init__(**kwargs)
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()
    
    
    def __len__(self):
        # Compute the number of batches to process per epoch.
        return int(np.ceil(len(self.image_paths) / self.batch_size))


    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Convert indexes to paths before passing them to __data_generation
        batch_paths = [self.image_paths[k] for k in indexes]
        X = self.__data_generation(batch_paths)

        for path in batch_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"File {path} not found.")

        # Logging for debugging
        if np.any(X == None):
            print(f"Found None values in batch {index}")

        
        return X, X # For an autoencoder, inputs and targets are the same


    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_paths):
        # Initialize X with the correct shape for hyperspectral images.
        X = np.empty((len(batch_paths), 128, 128, 224))
        
        
        for i, path in enumerate(batch_paths):
            try:
                with rasterio.open(path) as src:
                    # Adjusting read to match expected output dimensions directly if possible
                    image = src.read()  # This loads the image with shape (channels, height, width)
                    image = np.transpose(image, (1, 2, 0))  # Reorder dimensions to (height, width, channels)
                    image = image.astype(np.float32)  # Convert to float for normalization

                    assert image.shape == (128, 128, 224), f"Incorrect image shape for {path}: {image.shape}"
                    
                    if np.isnan(image).any():
                        print(f"NaN values detected in image loaded from {path}")
                        continue  # Consider handling this case more gracefully

                    if np.any(image == None):
                        print(f"None values detected in image loaded from {path}")
                        continue
                    
                    max_val = np.max(image)
                    if max_val == 0:
                        print(f"Image with only zero values detected: {path}")
                        continue  # Consider handling this case more gracefully
                    
                    # Normalizing the image
                    image /= (max_val + np.finfo(float).eps)
                     
                    X[i] = image  # Direct assignment without adding an extra dimension.

            except Exception as e:
                print(f"Error processing image {path}: {e}")
                    # Handling the error (e.g., by skipping this image) might require adjusting the shape of X or using a placeholder.
        
            # Asserting the final shapes to catch any inconsistencies.
            assert not np.any(np.isnan(X)), "NaN values found in X."
            assert not np.any(np.isinf(X)), "Inf values found in X."
    
        return X

