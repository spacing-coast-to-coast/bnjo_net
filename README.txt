## Project Setup:

1 - You will need to get spectral images from https://hyspecnet.rsim.berlin/. Follow the steps listed in the documentation for converting to .npy if using, otherwise use TIFF.
utils/tif_to_npy.py contains the code from the Jupyter notebooks the author's included with the study. Place the folder "hyspecnet-11k/patches" in the project's root.

2 - Specify the below options in main.py 
    MODEL_NAME = Name of model - string
    INPUT_FILE_TYPE =  Currently 'tif' and 'npy' accepted - string
    BATCH_SIZE = Size of image batches in sequence - int
    EPOSH = Number of epochs - int
    USE_CALLBACKS = [list of callbacks defined in setup_callbacks() in utils/utils.py]
        -'tf_logs': Sets up tensorboard logging: https://www.tensorflow.org/tensorboard/get_started
        -'keras_checkpoints': Sets up keras checkpoints. https://keras.io/api/callbacks/model_checkpoint/

## Development Tips:
- Try to keep any experimental stuff that might be useful in a notebook.
- Do not make changes directly to the 'Main' branch which will serve as the final version of the project. Branch off of 'development' then merge changes to back into 'development'. This will ensure only submission quality items are in 'Main'. 
- Once running successfully on a new machine, create a requirements.txt using ```pip freeze > requirements_<YOUR_MACHINE>.txt```