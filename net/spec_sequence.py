import tensorflow as tf
from glob import glob
import os
import numpy as np
from scipy.interpolate import interp1d
from net.spec_utils import parse_spectral_file, read_and_preprocess_spectral_data

class SpectralDataSequence(tf.keras.utils.Sequence):
    def __init__(self, file_paths, batch_size, target_wavelengths, shuffle=False):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.target_wavelengths = target_wavelengths
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.file_paths)
    
    def __len__(self):
        return np.ceil(len(self.file_paths) / self.batch_size).astype(int)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.file_paths)
            
    def __getitem__(self, idx):
        batch_file_paths = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = [read_and_preprocess_spectral_data(file_path, self.target_wavelengths) for file_path in batch_file_paths]
        batch_data = np.array(batch_data)
        batch_data = batch_data.reshape(-1, 224, 1)
        
        return batch_data, batch_data

def read_and_preprocess_spectral_file(file_path, target_wavelengths):
    metadata, wavelengths, reflectances = parse_spectral_file(file_path)
    wavelengths = np.array(wavelengths)
    reflectances = np.array(reflectances)
    interpolated_reflectances = interpolate_to_common_grid(wavelengths, reflectances, target_wavelengths)
    return interpolated_reflectances

def interpolate_to_common_grid(wavelengths, reflectances, target_wavelengths):
    interpolation_function = interp1d(wavelengths, reflectances, kind='linear', bounds_error=False, fill_value="extrapolate")
    interpolated_reflectances = interpolation_function(target_wavelengths)
    interpolated_reflectances = interpolated_reflectances.reshape(-1, 1)
    return interpolated_reflectances
