# Generative Models

## Introduction

This project is a part of 2024/2025 Summer Semester course in Deeplearning at Warsaw University of Technology, faculty of Mathematics and Information Sciences.

## Dataset

The dataset can be found in several different places in the internet, but we recommend using `fferlito/Cat-faces-dataset` repository.
To download it on linux:
```bash
wget https://github.com/fferlito/Cat-faces-dataset/raw/master/dataset-part1.tar.gz
wget https://github.com/fferlito/Cat-faces-dataset/raw/master/dataset-part2.tar.gz
wget https://github.com/fferlito/Cat-faces-dataset/raw/master/dataset-part3.tar.gz
```
Extract them to `cats/` directory:
```bash
mkdir -p cats/images
tar -xzvf dataset-part1.tar.gz -C cats/images --strip-components=1
tar -xzvf dataset-part2.tar.gz -C cats/images --strip-components=1
tar -xzvf dataset-part3.tar.gz -C cats/images --strip-components=1
```

## Dependencies

Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchsummary matplotlib

```

## Usage

Three different architectures were used to generate images:
* AutoEncoders
* Variational AutoEncoders (VAEs)
* Diffusion models
for each type of model separe branch is created.

## Results
