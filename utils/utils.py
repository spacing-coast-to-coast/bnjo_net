from pathlib import Path
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import datetime


def setup_model(model_name, use_callbacks=[]):
    dtm = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path("models") / f"{model_name}_{dtm}"

    # Create model directory
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model directory created: {model_dir}")

    # Setup callbacks based on the requested list
    callbacks = setup_callbacks(model_dir, use_callbacks)
    
    return str(model_dir), callbacks


def setup_callbacks(model_dir, use_callbacks):
    """Setup callbacks for the model training.
    Args:
        model_dir: The directory to save the model and logs.
        use_callbacks: A list of strings indicating which callbacks to use.

    Define more callbacks as needed here.
    """
    callbacks = []
    if 'tf_logs' in use_callbacks:
        logs_dir = model_dir / 'logs' / "tf_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(logs_dir))
        callbacks.append(tensorboard_callback)

    if 'keras_checkpoints' in use_callbacks:
        checkpoints_dir = model_dir / 'checkpoints'
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoints_dir / "best_model.keras"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), save_best_only=True, verbose=1)
        callbacks.append(checkpoint_callback)

    return callbacks


def pull_spectral_images(root_dir, type):
    root_path = Path(root_dir)
    if type == 'tif':
        print("Pulling spectral images from .TIF files")
        files = list(root_path.rglob("*SPECTRAL_IMAGE.TIF"))
    elif type == 'npy':
        print("Pulling spectral images from .npy files")
        files = list(root_path.rglob("*DATA.npy"))
    else:
        files = []

    return [str(file) for file in files], (128, 128, 224, 1) if type == 'tif' else (128, 128, 202, 1)


def save_model(model, model_name, model_dir):
    model_dir = Path(model_dir) 
    model_path = model_dir / f"{model_name}.keras"
    model.save(str(model_path))
    print(f"Model saved to {model_path}")


def load_my_model(model_path):
    return tf.keras.models.load_model(str(model_path))
