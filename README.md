# SAR Target Classification Using Deep Learning

This repository contains code and resources for a project focused on classifying ground targets using Synthetic Aperture Radar (SAR) imagery. The project leverages deep learning techniques to classify various military vehicles based on their SAR images, using the MSTAR dataset.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)

## Introduction

Synthetic Aperture Radar (SAR) is a form of radar used to create 2D or 3D images of objects, such as landscapes. This project aims to classify different military vehicles using SAR images from the MSTAR dataset. We utilize a Convolutional Neural Network (CNN) to identify the targets based on their SAR imagery.

## Dataset

The dataset used in this project is the MSTAR dataset, which contains SAR images of various military vehicles. The dataset is organized into folders representing different target types:

- 2S1
- BRDM_2
- BTR_60
- D7
- SLICY
- T62
- ZIL131
- ZSU_23_4

Each folder contains SAR images of the respective target.

## Project Structure

```plaintext
SAR_Target_Classification/
│
├── MSTAR_TargetImages/          # Dataset containing SAR images
│   ├── 2S1/
│   ├── BRDM_2/
│   ├── BTR_60/
│   ├── D7/
│   ├── SLICY/
│   ├── T62/
│   ├── ZIL131/
│   └── ZSU_23_4/
│
├── SAR_Target_Classification_Using_Deep_Learning.ipynb  # Main Jupyter Notebook
├── README.md                                            # This file
└── requirements.txt                                     # Python dependencies
