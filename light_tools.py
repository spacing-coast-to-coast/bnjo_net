import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def wavelength_to_band(bands, wavelength):
    bands_in_range = bands[(bands['CW (nm)'] - bands['FWHM (nm)']/2 <= wavelength) & 
                                 (bands['CW (nm)'] + bands['FWHM (nm)']/2 >= wavelength)]
    
    return bands_in_range[['BAND #', 'Type', 'CW (nm)', 'FWHM (nm)']]


def wavelength_to_bands(bands, wavelengths):
    bands_in_range = []
    for wavelength in wavelengths:
        bands_in_range.append(wavelength_to_band(bands, wavelength))
        
    return pd.concat(bands_in_range)


def bands_to_wavelengths(bands, bands_list=None):
    if bands_list is None:
        # If no specific bands list is provided, use all bands from the DataFrame
        bands_list = bands['BAND #'].tolist()
    
    band_info_list = []
    
    for band in bands_list:
        band_info = bands[bands['BAND #'] == band]
        
        if not band_info.empty:
            band_info_list.append({
                'BAND #': band,
                'Type': band_info['Type'].iloc[0],
                'CW (nm)': band_info['CW (nm)'].iloc[0],
                'FWHM (nm)': band_info['FWHM (nm)'].iloc[0]
            })
        else:
            band_info_list.append(None)
    
    band_info_list = pd.DataFrame(band_info_list)
    
    return band_info_list


def generate_spectral_profile(image, pixel_location=None):
    if pixel_location:
        # Extract the spectral profile for a specific pixel
        spectral_profile = image[pixel_location[0], pixel_location[0], :]
    else:
        # Average the spectral profile over the entire image
        spectral_profile = np.mean(np.mean(image, axis=0), axis=0)
    return spectral_profile

def plot_spectral_profile(band_info_df, wavelengths, spectral_profile):
    plt.figure(figsize=(14, 7))
    
    # Plot the spectral profile
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


def pull_spectral_images(root_dir):
    root_path = Path(root_dir)
    return [str(file) for file in root_path.rglob("*SPECTRAL_IMAGE.TIF")]


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