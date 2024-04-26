import tensorflow as tf
import os
import numpy as np


class HyperspectralSequence(tf.keras.utils.Sequence):
    def __init__(self, annotations, data_dir, selected_fruits, batch_size, max_bands=224, target_size=(64, 64), **kwargs):
        super().__init__(**kwargs)
        self.annotations = annotations
        self.data_dir = data_dir
        self.selected_fruits = selected_fruits
        self.batch_size = batch_size
        self.max_bands = max_bands
        self.indexes = self._filter_indexes()
        self.target_size = target_size  
        
    def _filter_indexes(self):
        return [i for i, record in enumerate(self.annotations['records']) if record['fruit'] in self.selected_fruits]
    
    def __len__(self):
        return np.ceil(len(self.indexes) / self.batch_size).astype(int)
    
    def _read_and_preprocess_hyperspectral_image(self, data_dir, data_file, hdr_file, max_bands):
        hdr_file_path = os.path.join(data_dir, hdr_file)
        data_file_path = os.path.join(data_dir, data_file)
        with open(hdr_file_path, "r") as f:
            header = {
                line.split("=")[0].strip(): line.split("=")[1].strip()
                for line in f
                if "=" in line
            }
        lines, samples, bands = (
            int(header["lines"]),
            int(header["samples"]),
            min(int(header["bands"]), max_bands),
        )
        with open(data_file_path, "rb") as f:
            data = np.fromfile(f, dtype=np.float32)[: lines * samples * bands].reshape(
                (lines, samples, bands)
            )
        
        data_resized = tf.image.resize(data, self.target_size).numpy()

        return data_resized
    
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_data = []
        
        for i in batch_indexes:
            record = self.annotations['records'][i]
            data = self._read_and_preprocess_hyperspectral_image(
                self.data_dir,
                record['files']['data_file'],
                record['files']['header_file'],
                self.max_bands
            )
            X_data.append(data)
        

        X_data_np = np.array(X_data, dtype=np.float32)
        X_data_np = np.expand_dims(X_data_np, axis=-1)
        
        return X_data_np, X_data_np
