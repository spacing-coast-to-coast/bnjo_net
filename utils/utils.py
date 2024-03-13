from pathlib import Path
import datetime
from tensorflow.keras.models import load_model
import os


def pull_spectral_images(root_dir, type):
    root_path = Path(root_dir)
    if type == 'tif':
        print("Pulling spectral images from .TIF files")
        return [str(file) for file in root_path.rglob("*SPECTRAL_IMAGE.TIF")], (128, 128, 224, 1)
    elif type == 'npy':
        print("Pulling spectral images from .npy files")
        return [str(file) for file in root_path.rglob("*DATA.npy")], (128, 128, 202, 1)


def save_model(model, model_name):
    dtm = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs('../models', exist_ok=True)
    model.save(f"../models/{model_name}_{dtm}.keras")
    print(f"Model saved as {model_name}_{dtm}.keras")

def get_model(model_name):
    return load_model(model_name)