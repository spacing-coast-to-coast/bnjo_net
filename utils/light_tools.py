import pandas as pd
import numpy as np


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


