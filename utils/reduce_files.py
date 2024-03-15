from pathlib import Path
import shutil

def get_directory_size(directory):
    """Calculate the total size of all files in the directory."""
    total_size = sum(file.stat().st_size for file in directory.rglob('*') if file.is_file())
    return total_size

def format_size(size):
    """Format the size for human-readable output."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"

def copy_files_to_new_dir(root_dir, target_dir):
    root_path = Path(root_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)  # Ensure the target directory exists

    # List all files matching the criteria
    spectral_images = root_path.rglob("*SPECTRAL_IMAGE.TIF")
    data_files = root_path.rglob("*DATA.npy")

    files = list(spectral_images) + list(data_files)

    print(f"Copying {len(files)} files to {target_path}")

    # Function to copy files to the target directory with handling for duplicate filenames
    def copy_files(files):
        for file_to_copy in files:
            # Construct the target file path
            target_file_path = target_path / file_to_copy.name
            # If the file already exists, append a number to the filename to avoid overwriting
            counter = 1
            while target_file_path.exists():
                target_file_path = target_path / f"{file_to_copy.stem}_{counter}{file_to_copy.suffix}"
                counter += 1
            shutil.copy(file_to_copy, target_file_path)

    # Copy all matching files to the target directory
    copy_files(files)

    # After copying, calculate and print the sizes
    root_size = get_directory_size(root_path)
    target_size = get_directory_size(target_path)
    print(f"Done copying files.\nRoot directory size: {format_size(root_size)}\nTarget directory size: {format_size(target_size)}")


root_dir = '../hyspecnet-11k/patches'
target_dir = '../hyspecnet-11k/patches_tif_npy/'
copy_files_to_new_dir(root_dir, target_dir)
