Directory structure:

Put DeepHS and ecostress into their own directories
DeepHS fruit dirs should be unzipped and placed in DeepHS.
Create a folder for models to save and load. This structure is designed
to be used with any dataset.

|--Directory Structure
|(root)
|
|-DeepHS/
|--annotations
|--Avacado
|--Kaki
|--Kiwi
|--Mango
|--Papaya
|
|-ecospeclib-all/
|
|-HySpecNet or other HSI images
|--hyspecnet-11k
|--hyspecnet-tools
|
|-models/
|
|-net/
|
|-main.ipynb


DeepHS: 
Paper: https://arxiv.org/abs/2104.09808v1
Dataset: https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/
Citation:
# @INPROCEEDINGS{10283385,
#   author={Fuchs, Martin Hermann Paul and Demir, Begüm},
#   booktitle={IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium}, 
#   title={HySpecNet-11k: a Large-Scale Hyperspectral Dataset for Benchmarking Learning-Based Hyperspectral Image Compression Methods}, 
#   year={2023},
#   volume={},
#   number={},
#   pages={1779-1782},
#   doi={10.1109/IGARSS52108.2023.10283385}}

HySpecNet:
Paper and Dataset: https://hyspecnet.rsim.berlin/
Citation:
@inproceedings{Varga2021,
abstract = {We present a system to measure the ripeness of fruit with a hyperspectral camera and a suitable deep neural network architecture. This architecture did outperform competitive baseline models on the prediction of the ripeness state of fruit. For this, we recorded a data set of ripening avocados and kiwis, which we make public. We also describe the process of data collection in a manner that the adaption for other fruit is easy. The trained network is validated empirically, and we investigate the trained features. Furthermore, a technique is introduced to visualize the ripening process.},
archivePrefix = {arXiv},
arxivId = {2104.09808},
author = {Varga, Leon Amadeus and Makowski, Jan and Zell, Andreas},
booktitle = {2021 International Joint Conference on Neural Networks (IJCNN)},
doi = {10.1109/IJCNN52387.2021.9533728},
eprint = {2104.09808},
isbn = {978-1-6654-3900-8},
keywords = {Index Terms-hyperspectral,convolutional neu-ral network,deep learning,ripening fruit},
month = {jul},
pages = {1--8},
publisher = {IEEE},
title = {{Measuring the Ripeness of Fruit with Hyperspectral Imaging and Deep Learning}},
url = {https://arxiv.org/abs/2104.09808v1 http://arxiv.org/abs/2104.09808 https://ieeexplore.ieee.org/document/9533728/},
year = {2021}
}

preprocessing steps include stripping all files that are not .jpg and .npy