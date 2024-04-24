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

# @INPROCEEDINGS{10283385,
#   author={Fuchs, Martin Hermann Paul and Demir, Begüm},
#   booktitle={IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium}, 
#   title={HySpecNet-11k: a Large-Scale Hyperspectral Dataset for Benchmarking Learning-Based Hyperspectral Image Compression Methods}, 
#   year={2023},
#   volume={},
#   number={},
#   pages={1779-1782},
#   doi={10.1109/IGARSS52108.2023.10283385}}

#preprocessing steps include stripping all files that are not .jpg and .npy