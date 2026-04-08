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

This project implements four different architectures for image generation:

  * Autoencoders
  * Variational Autoencoders (VAEs)
  * Diffusion Models
  * GAN

Each model architecture is maintained in a separate branch within this repository.

## Results

### Generated Image Samples

Below are samples of the generated images.

#### Autoencoders

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

#### VAEs

<div align="center">
<table\>
<td\><img src="generated/VAEs/VAE0_readme.png" width="50" alt="Cat image"></td\>
<td\><img src="generated/VAEs/VAE1_readme.png" width="50" alt="Cat image"></td\>
<td\><img src="generated/VAEs/VAE2_readme.png" width="50" alt="Cat image"></td\>
<td\><img src="generated/VAEs/VAE3_readme.png" width="50" alt="Cat image"></td\>
<td\><img src="generated/VAEs/VAE4_readme.png" width="50" alt="Cat image"></td\>
<td\><img src="generated/VAEs/VAE5_readme.png" width="50" alt="Cat image"></td\>
<td\><img src="generated/VAEs/VAE6_readme.png" width="50" alt="Cat image"></td\>
<td\><img src="generated/VAEs/VAE7_readme.png" width="50" alt="Cat image"></td\>
</table>
</div>

#### Diffusion

<div align="center"\>
<table\>
<td\><img src="generated/Diffusion/diffusion0_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/Diffusion/diffusion1_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/Diffusion/diffusion2_readme.png" width="50" alt="Cat image"\></td\>
<td\><img src="generated/Diffusion/diffusion3_readme.png" width="50" alt="Cat image"\></td\>
</table\>
</div>

#### GAN

<div align="center">
<img src="generated/GAN/GAN0_readme.png" width="200" alt="Cat image"\>
<img src="generated/GAN/GAN1_readme.png" width="200" alt="Cat image"\>
</div>
