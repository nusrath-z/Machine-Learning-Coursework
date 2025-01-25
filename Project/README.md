## Introduction to Machine Learning - Course Project

This folder contains the implementation and results for the **Introduction to Machine Learning** course project working with a team of 5 members. The project focuses on applying deep learning techniques, specifically Convolutional Neural Networks (CNN), to image classification tasks using real-world datasets. The project includes both training and fine-tuning models, as well as exploring the concept of **Transfer Learning** to improve model performance on different datasets.

## Project Overview

The primary objective of this project is to:
- Train and tune CNN models on datasets from the fields of **computational pathology** and **computer vision**.
- Analyze and visualize the performance of the models, including using dimensionality reduction techniques like **t-SNE**.
- Explore the concept of **Transfer Learning**, where previously trained models on one dataset are tested on datasets from different applications.

### Datasets:
1. **Colorectal Cancer Classification**: 6K image patches split into 3 classes: Smooth Muscle (MUS), Normal Colon Mucosa (NORM), and Cancer-associated Stroma (STR).
2. **Prostate Cancer Classification**: 6K image patches for classifying Prostate Cancer Tumor Tissue, Benign Glandular Prostate Tissue, and Benign Non-Glandular Prostate Tissue.
3. **Animal Faces Classification**: 6K images of Cats, Dogs, and Wildlife Animals.

### Project Tasks:
1. **Task 1**:
   - Train a CNN model- ResNet18 was used, on the **Colorectal Cancer Classification** dataset.
   - Use **t-SNE** for dimensionality reduction and visualize the CNN’s feature extraction outputs.
   
2. **Task 2**:
   - Apply the final CNN model (from Task 1) on the **Prostate Cancer** and **Animal Faces** datasets for feature extraction.
   - Visualize the extracted features using **t-SNE**.
   - Apply a classical machine learning algorithm, LR, and RF were used for the classification on the extracted features from the datasets.


### Technologies & Tools:
- **Machine Learning & Deep Learning**: CNNs, Transfer Learning, Feature Extraction, t-SNE, LR, Random Forest
- **Libraries**: PyTorch, scikit-learn, Matplotlib, Pandas, NumPy
- **Techniques**: Dimensionality Reduction, Model Training, Feature Visualization, Model Evaluation

  ## Results:
- **Training Accuracy/Loss Curves**: Visualizations of training accuracy and loss for each model.
- **t-SNE Visualizations**: Feature embeddings visualized using t-SNE for the datasets.
- **Classification Accuracy Metrics**: The performance evaluation metrics for each dataset, including accuracy scores and confusion matrices.

## Project Team:
- **Mhd Eyad**
- **Hisham**
- **Omar**
- **Nour**
- **Nusrath**

