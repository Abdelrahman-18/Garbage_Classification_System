# ♻️ Garbage Classification System

## 📌 Project Overview

Garbage classification is an important computer vision application that supports smarter recycling and waste management systems.  
This project focuses on classifying garbage images into multiple categories using Deep Learning and Computer Vision techniques with PyTorch.

The system:

- Processes garbage image datasets
- Applies image preprocessing and augmentation
- Uses transfer learning with a pretrained CNN
- Trains and evaluates the classification model
- Visualizes model performance with metrics and confusion matrices
- Deploys the trained model through a GUI application

---

## 📋 Pipeline Summary

| Step | Description |
|------|-------------|
| 0 | Configuration → Define dataset paths, image size, and hyperparameters |
| 1 | Data Loading → Load and preprocess garbage images |
| 2 | Data Augmentation → Apply transformations to improve generalization |
| 3 | Dataset Split → Train / Validation / Test preparation |
| 4 | Model Building → Transfer learning using MobileNet |
| 5 | Model Training → Train the CNN model on garbage categories |
| 6 | Evaluation → Accuracy, classification report, and confusion matrix |
| 7 | Model Saving → Export trained model weights |
| 8 | GUI Deployment → Real-time image prediction application |

---

## 🧠 Deep Learning Architecture

The project uses a pretrained **MobileNet** architecture for image classification.

### Key Components

### Transfer Learning
A pretrained CNN is used to improve learning efficiency and achieve higher accuracy with limited data.

### Data Augmentation
Image transformations such as flipping, rotation, and normalization are applied to improve model robustness.

### Feature Learning
Instead of handcrafted features, the CNN automatically learns important visual patterns such as:

- Shapes
- Edges
- Textures
- Object structures

---

## 🤖 Garbage Categories

The model classifies garbage into multiple categories, including:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

---

## 🔧 Training Strategy

The model training process includes:

- Transfer Learning
- Data Augmentation
- Cross Entropy Loss
- Adam Optimizer
- Validation-based model evaluation

### Tuned Parameters

Example tuned hyperparameters:

- Learning Rate
- Batch Size
- Number of Epochs
- Optimizer Configuration

The best configuration is selected based on validation accuracy.

---

## 📊 Evaluation

Final evaluation is performed on a held-out test dataset using:

- Accuracy
- Classification Report
- Confusion Matrix
- Per-class Performance Analysis

### Example Results

- High classification accuracy across major garbage categories
- Strong confusion matrix diagonal dominance indicating reliable predictions

---

## 🌐 GUI Deployment

The trained model is deployed through a desktop GUI application using Tkinter.

### GUI Features

- Upload an image
- Automatic image preprocessing
- Real-time garbage classification
- Instant prediction display

This allows users to interact with the trained model without running notebook code manually.

---

## 🖼️ Project Preview

<!-- Add project screenshots here -->

<p align="center">
  <img width="1126" height="912" alt="Image" src="https://github.com/user-attachments/assets/665920fb-4dd3-4ac9-bc9d-e10889e0b3c8" />
</p>

---

## 📂 Dataset Structure

```bash
Dataset/
│
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Run the notebook:

```bash
jupyter notebook Garbage_Classifier.ipynb
```

Run the GUI application:

```bash
python GUI.py
```

---

## 📈 Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Seaborn
- Tkinter


