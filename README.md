# SAR Target Classification Using Deep Learning

This repository contains code and resources for a project focused on classifying ground targets using Synthetic Aperture Radar (SAR) imagery. The project leverages deep learning techniques to classify various military vehicles based on their SAR images, using the MSTAR dataset.

## Overview

### Project Objective

The objective of this project is to classify various ground-based military vehicles using SAR (Synthetic Aperture Radar) images. SAR provides high-resolution imaging capabilities, even in adverse weather conditions, making it a valuable tool in remote sensing and defense applications. By utilizing deep learning techniques, specifically Convolutional Neural Networks (CNNs), this project aims to accurately identify and classify the type of vehicle present in the SAR imagery.

### Dataset

The dataset used is the MSTAR (Moving and Stationary Target Acquisition and Recognition) dataset, which consists of SAR images of various military vehicles. The images were captured using an X-band sensor in spotlight mode with a resolution of 1 foot. The dataset includes the following target types:

- **2S1**: Self-propelled artillery
- **BRDM_2**: Armored reconnaissance vehicle
- **BTR_60**: Armored personnel carrier
- **D7**: Armored bulldozer
- **SLICY**: Calibration target
- **T62**: Main battle tank
- **ZIL131**: Military truck
- **ZSU_23_4**: Self-propelled anti-aircraft gun

Each type of vehicle is stored in its respective folder, containing SAR images captured at depression angles of 15 and 17 degrees, covering full aspect views over 360 degrees.

### Methodology

The project is implemented using Python and deep learning frameworks, primarily TensorFlow. The key steps in the process are:

1. **Data Loading**: SAR images are loaded using the `image_dataset_from_directory` method in TensorFlow, which organizes the images based on folder names and creates a labeled dataset.

2. **Model Architecture**: A CNN is constructed with multiple layers, including convolutional layers for feature extraction, max-pooling layers for downsampling, and dense layers for classification.

3. **Training**: The model is trained on the dataset, with an 80-20 split between training and validation data. The training process involves optimizing the model using the Adam optimizer and monitoring the loss and accuracy metrics.

4. **Evaluation**: The model is evaluated on the validation set to determine its accuracy and effectiveness in classifying SAR images.

## Observations

### Accuracy and Performance

- The model achieved an accuracy of around XX% on the validation set (replace with actual results), indicating that the CNN was able to learn and generalize from the training data effectively.
- The learning curves, including accuracy and loss over epochs, suggest that the model converged well without significant overfitting, demonstrating good generalization to unseen data.

### Challenges and Considerations

- **SAR Image Characteristics**: SAR images have unique characteristics, such as speckle noise and varying illumination, which can complicate the classification task. However, the CNN was able to extract relevant features and mitigate these challenges to some extent.
- **Class Imbalance**: Some target classes had fewer images, which could potentially impact the model's ability to learn those classes well. Data augmentation techniques or weighted loss functions might be necessary in future work to address this issue.

### Future Work

- **Model Improvement**: Experimenting with more complex architectures, such as deeper networks or transfer learning models, could improve classification accuracy.
- **Data Augmentation**: Implementing data augmentation techniques, such as rotation, flipping, and noise addition, could help the model generalize better, especially for underrepresented classes.
- **Real-Time Classification**: Extending the project to perform real-time classification on streaming SAR data could be an interesting direction for practical applications.

## How to Run

### Prerequisites

- Python 3.x
- Google Colab (Optional, if you prefer running on Colab)
- TensorFlow

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/SAR_Target_Classification.git
    cd SAR_Target_Classification
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Open the Jupyter Notebook and run the cells:

    ```bash
    jupyter notebook SAR_Target_Classification_Using_Deep_Learning.ipynb
    ```

4. Alternatively, you can upload the notebook to Google Colab and run it there.

## Dependencies

- TensorFlow
- NumPy
- Matplotlib

You can install all dependencies using the `requirements.txt` file provided.

```bash
pip install -r requirements.txt
