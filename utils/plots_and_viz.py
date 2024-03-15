import matplotlib.pyplot as plt
import numpy as np

def plot_spectral_profile(band_info_df, wavelengths, spectral_profile):
    plt.figure(figsize=(14, 7))
    
    # Plot spectral profile
    plt.plot(wavelengths, spectral_profile, label='Spectral Profile', color='blue', linewidth=2)
    
    ticks = []
    labels = []
    for i, (index, row) in enumerate(band_info_df.iterrows()):
        if i % 5 == 0:  # Select every 5th band for clarity
            plt.axvline(x=row['CW (nm)'], color='red', linestyle='--', linewidth=1)
            ticks.append(row['CW (nm)'])
            labels.append(f"Band {row['BAND #']}")
            

    plt.xticks(ticks, labels, rotation='vertical') 
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)
    plt.title('Spectral Profile with Highlighted Bands', fontsize=14)
    plt.legend()
    plt.tight_layout() 
    plt.show()


def visualize_bands(image, title="Spectral Band"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    band_indices = [0, image.shape[2] // 2, image.shape[2] - 1]  # First, middle, and last band
    
    for ax, band_index in zip(axes, band_indices):
        ax.imshow(image[:, :, band_index], cmap='gray')
        ax.set_title(f'{title} {band_index + 1}')
    plt.show()


def plot_band_histograms(image, band_indices):
    fig, axes = plt.subplots(1, len(band_indices), figsize=(15, 5))
    for ax, band_index in zip(axes, band_indices):
        ax.hist(image[:, :, band_index].ravel(), bins=256, color='blue', alpha=0.7)
        ax.set_title(f'Band {band_index + 1} Histogram')
    plt.tight_layout()
    plt.show()

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

    # Filter out images and bands with 0% NaNs
    nan_info_filtered = {
        image_idx: {band_idx: nan_pct for band_idx, nan_pct in bands.items() if nan_pct > 0}
        for image_idx, bands in nan_info.items() if any(nan_pct > 0 for nan_pct in bands.values())
    }

    return nan_info_filtered, global_min, global_max
