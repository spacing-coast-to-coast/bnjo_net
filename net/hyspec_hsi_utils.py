import pandas as pd
import os

def load_hyspecnet_images(directory):
    training_csv_file = f"{directory}/hyspecnet-11k/splits/easy/train.csv"
    validation_csv_file = f"{directory}/hyspecnet-11k/splits/easy/val.csv"
    test_csv_file = f"{directory}/hyspecnet-11k/splits/easy/test.csv"

    base_image_directory = f"{directory}/hyspecnet-11k/patches"

    training_df = pd.read_csv(training_csv_file)
    validation_df = pd.read_csv(validation_csv_file)
    test_df = pd.read_csv(test_csv_file)

    training_paths = [os.path.join(base_image_directory, path) for path in training_df.iloc[:, 0].tolist()]
    validation_paths = [os.path.join(base_image_directory, path) for path in validation_df.iloc[:, 0].tolist()]
    test_paths = [os.path.join(base_image_directory, path) for path in test_df.iloc[:, 0].tolist()]

    return training_paths, validation_paths, test_paths

