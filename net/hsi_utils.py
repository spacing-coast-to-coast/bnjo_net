import json
import numpy as np


def load_DeepHS_images(directory):
    training_annotations_file = f"{directory}/annotations/train_only_labeled_v2.json"
    validation_annotations_file = f"{directory}/annotations/val_v2.json"
    test_annotations_file = f"{directory}/annotations/test_v2.json"

    with open(training_annotations_file) as f:
        training_annotations = json.load(f)
    with open(validation_annotations_file) as f:
        validation_annotations = json.load(f)
    with open(test_annotations_file) as f:
        test_annotations = json.load(f)

    print("Datasets loaded:")
    print(
        "Training annotations:",
        training_annotations["records"][0].keys(),
        len(training_annotations["records"]),
    )
    print(
        "Validation annotations:",
        validation_annotations["records"][0].keys(),
        len(validation_annotations["records"]),
    )
    print(
        "Test annotations:",
        test_annotations["records"][0].keys(),
        len(test_annotations["records"]),
    )

    return training_annotations, validation_annotations, test_annotations


def read_and_visualize_hyperspectral_image(
    data_dir, data_file, hdr_file, max_bands, rgb_bands=(29, 19, 9)
):
    hdr_file_path = f"{data_dir}/{hdr_file}"
    data_file_path = f"{data_dir}/{data_file}"

    with open(hdr_file_path, "r") as f:
        header = {
            line.split("=")[0].strip(): line.split("=")[1].strip()
            for line in f
            if "=" in line
        }

    lines, samples, bands = (
        int(header["lines"]),
        int(header["samples"]),
        min(int(header["bands"]), max_bands),
    )

    with open(data_file_path, "rb") as f:
        data = np.fromfile(f, dtype=np.float32)[: lines * samples * bands].reshape(
            (lines, samples, bands)
        )

    rgb_example = np.stack(
        [data[:, :, rgb_bands[0]], data[:, :, rgb_bands[1]], data[:, :, rgb_bands[2]]],
        axis=-1,
    )
  
    rgb_example = np.clip(rgb_example / np.max(rgb_example), 0, 1)

    return data, rgb_example



