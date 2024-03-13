import numpy as np
import matplotlib.pyplot as plt

def analyze_hyperspectral_data(images, plot_nans=False):
    """
    Analyze hyperspectral images for NaN values and find global min/max values.
    
    Parameters:
    - images: A list of hyperspectral images, each with dimensions (height, width, bands).
    - plot_nans: Whether to plot the locations of NaN values.
    
    Returns a tuple containing:
    - A dictionary with NaN information.
    - The global minimum value.
    - The global maximum value.
    """
    global_min = float('inf')
    global_max = float('-inf')
    nan_info = {}

    for i, image in enumerate(images):
        image_min = np.nanmin(image)
        image_max = np.nanmax(image)
        global_min = min(global_min, image_min)
        global_max = max(global_max, image_max)
        
        nan_info[i] = {}
        for band_index in range(image.shape[2]):
            band_data = image[:, :, band_index]
            nan_mask = np.isnan(band_data)
            nan_percentage = np.sum(nan_mask) / band_data.size * 100
            nan_info[i][band_index] = nan_percentage

            if plot_nans and nan_percentage > 0:
                plt.imshow(nan_mask, cmap='hot')
                plt.title(f'Image {i} - NaN Locations in Band {band_index}')
                plt.colorbar()
                plt.show()

    # Filter out images and bands with 0% NaNs for a concise report
    nan_info_filtered = {
        image_idx: {band_idx: nan_pct for band_idx, nan_pct in bands.items() if nan_pct > 0}
        for image_idx, bands in nan_info.items() if any(nan_pct > 0 for nan_pct in bands.values())
    }

    return nan_info_filtered, global_min, global_max
