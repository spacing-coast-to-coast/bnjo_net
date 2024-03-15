import numpy as np

def normalize_image(image, target_range=(-1, 1)):
    """
    Normalize the image data to a specified range.

    Parameters:
    - image: numpy array, the image data to be normalized.
    - target_range: tuple, the target range for normalization. Default is (-1, 1).

    Returns:
    - Normalized image data.
    """
    # Extract the min and max values from the target range
    min_target, max_target = target_range
    
    # Compute the actual min and max from the image
    min_val, max_val = np.min(image), np.max(image)
    
    # Normalize to [0, 1]
    image = (image - min_val) / (max_val - min_val)
    
    if target_range == (-1, 1):
        # Scale to [-1, 1]
        image = 2 * image - 1
    elif target_range != (0, 1):
        # Adjust for a custom range [min_target, max_target]
        image = image * (max_target - min_target) + min_target
    
    return image
