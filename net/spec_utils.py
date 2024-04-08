import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


def parse_spectral_file(file_path, spectra_files_dir):
    full_path = os.path.join(spectra_files_dir, os.path.basename(file_path))
    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        # if UTF-8 doesn't work
        with open(full_path, 'r', encoding='ISO-8859-1') as file:
            content = file.read()

    lines = content.strip().split('\n')
    metadata = {}
    spectra = []
    parse_metadata_flag = False

    for line in lines:
        if not parse_metadata_flag:
            if line.strip() == "":
                parse_metadata_flag = True
            else:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
        else:
            wavelengths, reflectances = line.split()
            spectra.append((float(wavelengths), float(reflectances)))

    wavelengths, reflectances = zip(*spectra)

    return metadata, list(wavelengths), list(reflectances)


def read_and_preprocess_spectral_data(directory_path, target_wavelengths=np.linspace(1.0, 2.5, 224)):
    all_spectral_data = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.spectrum.txt'):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                metadata, wavelengths, reflectances = parse_spectral_file(file_path, directory_path)
                if wavelengths:
                    all_spectral_data.append({
                        'filename': filename,
                        'metadata': metadata,
                        'wavelengths': wavelengths,
                        'reflectances': reflectances
                    })

    # Apply preprocessing steps
    print(f"Found {len(all_spectral_data)} spectral files")
    normalized_spectral_data = normalize_reflectances(all_spectral_data)
    smoothed_spectral_data = smooth_spectra(normalized_spectral_data)
    interpolated_spectral_data = interpolate_to_common_grid(smoothed_spectral_data, target_wavelengths)

    
    reflectance_values = np.array([spectrum['reflectances'] for spectrum in interpolated_spectral_data])
    scaler = StandardScaler()
    X_data_normalized = scaler.fit_transform(reflectance_values)

    return X_data_normalized

# def parse_spectral_file(file_path):
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             content = file.read()
#     except UnicodeDecodeError:
#         # Try a different encoding if UTF-8 doesn't work
#         with open(file_path, 'r', encoding='ISO-8859-1') as file:
#             content = file.read()

#     lines = content.strip().split('\n')
#     metadata = {}
#     spectra = []
#     parse_metadata_flag = False

#     for line in lines:
#         if not parse_metadata_flag:
#             if line.strip() == "":
#                 parse_metadata_flag = True
#             else:
#                 key, value = line.split(':', 1)
#                 metadata[key.strip()] = value.strip()
#         else:
#             wavelengths, reflectances = line.split()
#             spectra.append((float(wavelengths), float(reflectances)))

#     wavelengths, reflectances = zip(*spectra)

#     return metadata, list(wavelengths), list(reflectances)

def read_and_preprocess_spectral_data_single(file_or_directory_path, target_wavelengths=np.linspace(1.0, 2.5, 224), is_directory=False):
    all_spectral_data = []

    if is_directory:
        filenames = [f for f in os.listdir(file_or_directory_path) if f.endswith('.spectrum.txt')]
    else:
        filenames = [os.path.basename(file_or_directory_path)]

    for filename in filenames:
        file_path = os.path.join(file_or_directory_path if is_directory else os.path.dirname(file_or_directory_path), filename)
        metadata, wavelengths, reflectances = parse_spectral_file(file_path)
        if wavelengths:
            all_spectral_data.append({
                'filename': filename,
                'metadata': metadata,
                'wavelengths': wavelengths,
                'reflectances': reflectances
            })

    print(f"Found {len(all_spectral_data)} spectral files")
    normalized_spectral_data = normalize_reflectances(all_spectral_data)
    smoothed_spectral_data = smooth_spectra(normalized_spectral_data)
    interpolated_spectral_data = interpolate_to_common_grid(smoothed_spectral_data, target_wavelengths)

    reflectance_values = np.array([spectrum['reflectances'] for spectrum in interpolated_spectral_data])
    scaler = StandardScaler()
    X_data_normalized = scaler.fit_transform(reflectance_values)

    return X_data_normalized

def normalize_reflectances(spectra):
    normalized_spectra = []
    for spectrum in spectra:
        reflectances = np.array(spectrum['reflectances'])
        normalized_reflectances = (reflectances - np.min(reflectances)) / (np.max(reflectances) - np.min(reflectances))
        spectrum['reflectances'] = normalized_reflectances.tolist()
        normalized_spectra.append(spectrum)
    return normalized_spectra

def smooth_spectra(spectra):
    smoothed_spectra = []
    for spectrum in spectra:
        wavelengths = np.array(spectrum['wavelengths'])
        reflectances = np.array(spectrum['reflectances'])
        smoothed_reflectances = savgol_filter(reflectances, window_length=5, polyorder=3)
        spectrum['reflectances'] = smoothed_reflectances.tolist()
        smoothed_spectra.append(spectrum)
    return smoothed_spectra

def interpolate_to_common_grid(spectra, target_wavelengths):
    interpolated_spectra = []
    for spectrum in spectra:
        current_wavelengths = np.array(spectrum['wavelengths'])
        current_reflectances = np.array(spectrum['reflectances'])
        interpolation_function = interp1d(current_wavelengths, current_reflectances, kind='linear', bounds_error=False, fill_value="extrapolate")
        interpolated_reflectances = interpolation_function(target_wavelengths)
        spectrum['wavelengths'] = target_wavelengths.tolist()
        spectrum['reflectances'] = interpolated_reflectances.tolist()
        interpolated_spectra.append(spectrum)
    return interpolated_spectra

def apply_pca(X_data_normalized, n_components=0.95):
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X_data_normalized)
    variance_explained = np.sum(pca.explained_variance_ratio_)
    return X_transformed, variance_explained
