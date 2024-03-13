import glob
import multiprocessing
import numpy as np
import rasterio
from multiprocessing import freeze_support


in_directory = "../hyspecnet-11k/patches/"

invalid_channels = [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 160, 161, 162, 163, 164, 165, 166]
valid_channels_ids = [c+1 for c in range(224) if c not in invalid_channels]

minimum_value = 0
maximum_value = 10000

in_patches = glob.glob(f"{in_directory}**/**/*SPECTRAL_IMAGE.TIF")

def convert(patch_path):
    # load patch
    dataset = rasterio.open(patch_path)
    # remove nodata channels
    src = dataset.read(valid_channels_ids)
    # clip data to remove uncertainties
    clipped = np.clip(src, a_min=minimum_value, a_max=maximum_value)
    # min-max normalization
    out_data = (clipped - minimum_value) / (maximum_value - minimum_value)
    out_data = out_data.astype(np.float32)
    # save npy
    out_path = patch_path.replace("SPECTRAL_IMAGE", "DATA").replace("TIF", "npy")
    np.save(out_path, out_data)

if __name__ == '__main__':
    freeze_support()
    with multiprocessing.Pool(64) as pool:
        pool.map(convert, in_patches)


# @INPROCEEDINGS{10283385,
#   author={Fuchs, Martin Hermann Paul and Demir, Begüm},
#   booktitle={IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium}, 
#   title={HySpecNet-11k: a Large-Scale Hyperspectral Dataset for Benchmarking Learning-Based Hyperspectral Image Compression Methods}, 
#   year={2023},
#   volume={},
#   number={},
#   pages={1779-1782},
#   doi={10.1109/IGARSS52108.2023.10283385}}
