# Generative Models

## Introduction

This repository contains a project developed for the Deep Learning course during the 2024/2025 Summer Semester at the Warsaw University of Technology, Faculty of Mathematics and Information Science.

## Dataset

The dataset can be obtained from various sources, but we recommend using the `fferlito/Cat-faces-dataset` repository.

To download the dataset on Linux, run the following commands:

```bash
wget [https://github.com/fferlito/Cat-faces-dataset/raw/master/dataset-part1.tar.gz](https://github.com/fferlito/Cat-faces-dataset/raw/master/dataset-part1.tar.gz)

wget [https://github.com/fferlito/Cat-faces-dataset/raw/master/dataset-part2.tar.gz](https://github.com/fferlito/Cat-faces-dataset/raw/master/dataset-part2.tar.gz)

wget [https://github.com/fferlito/Cat-faces-dataset/raw/master/dataset-part3.tar.gz](https://github.com/fferlito/Cat-faces-dataset/raw/master/dataset-part3.tar.gz)
````

Extract the downloaded archives into the `cats/images` directory:

```bash
mkdir -p cats/images
tar -xzvf dataset-part1.tar.gz -C cats/images --strip-components=1
tar -xzvf dataset-part2.tar.gz -C cats/images --strip-components=1
tar -xzvf dataset-part3.tar.gz -C cats/images --strip-components=1
```

## Dependencies

To set up the environment on Linux, create a virtual environment and install the required packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchsummary matplotlib
```

## Usage

### 1. Prepare the Directory

First, create the necessary directory to store the trained model weights:

```bash
mkdir -p models/AutoEncoders
mkdir -p generated/AutoEncoder
```

### 2. Train the Model

Run the `train.py` script to train the autoencoder. The script will save the model automatically and display examples of image reconstruction during the process. 

* To interrupt the training process at any time, press `Ctrl+C` in your terminal. 
* Feel free to explore the code and experiment with different architectural choices.

### 3. Generate Images

Once the model is successfully trained, execute the `generate.py` script to produce new images. By default, the generated outputs will be saved in the `generated/AutoEncoders/` directory.

## Results

### Generated Image Samples

Below are samples of the generated images.

> **Note:** Please note that all images generated using the standard Autoencoder exhibit a distinct blue artifact in the center. The root cause of this phenomenon requires further investigation.

<div align="center">
<table\>
<td\><img src="generated/AutoEncoder/AutoEncoder0_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/AutoEncoder/AutoEncoder1_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/AutoEncoder/AutoEncoder2_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/AutoEncoder/AutoEncoder3_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/AutoEncoder/AutoEncoder4_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/AutoEncoder/AutoEncoder5_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/AutoEncoder/AutoEncoder6_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/AutoEncoder/AutoEncoder7_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/AutoEncoder/AutoEncoder8_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/AutoEncoder/AutoEncoder9_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/AutoEncoder/AutoEncoder10_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/AutoEncoder/AutoEncoder11_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/AutoEncoder/AutoEncoder12_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/AutoEncoder/AutoEncoder13_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/AutoEncoder/AutoEncoder14_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/AutoEncoder/AutoEncoder15_readme.png" width="50" alt="Cat image"\></td\>
</table>
</div>
