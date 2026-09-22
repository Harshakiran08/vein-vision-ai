# Vein Vision AI

> Deep learning-based medical image segmentation for detecting and visualizing vein structures from skin images using U-Net and R2U-Net.

## Overview

**Vein Vision AI** is a computer vision and deep learning project that performs **vein segmentation from medical skin images**.

The system uses semantic segmentation models to identify vein regions and generate segmentation masks that make the detected structures easier to visualize.

The project explores the application of **deep learning, image processing, and biomedical image analysis** using U-Net and R2U-Net architectures.

> **Research / educational project:** This project is intended for experimentation and research in medical image segmentation and is not a medical diagnostic or clinical decision-making system.

---

## Key Features

- Vein segmentation from input images
- U-Net implementation for semantic segmentation
- R2U-Net implementation with recurrent residual blocks
- Image preprocessing and normalization
- Dataset preparation and loading
- Model training and evaluation
- Segmentation prediction
- Output visualization
- Modular Python-based project structure

---

## Tech Stack

### Programming Language

`Python`

### Deep Learning

`TensorFlow` `Keras`

### Computer Vision & Data Processing

`OpenCV` `NumPy` `Matplotlib` `Scikit-learn`

### Models

`U-Net` `R2U-Net`

---

## How It Works

```text
Input Image
     │
     ▼
Image Preprocessing
     │
     ├── Resize
     ├── Normalize
     └── Prepare Image
     │
     ▼
Deep Learning Model
     │
     ├── U-Net
     │
     └── R2U-Net
     │
     ▼
Vein Segmentation
     │
     ▼
Segmentation Mask
     │
     ▼
Visualization
```

---

## Model Architectures

### U-Net

U-Net is an encoder-decoder architecture widely used for biomedical image segmentation.

The encoder extracts hierarchical image features, while the decoder reconstructs the spatial representation. Skip connections between the encoder and decoder help preserve important spatial information.

### R2U-Net

R2U-Net extends the U-Net architecture by incorporating **recurrent convolutional operations and residual connections**.

The architecture is designed to improve feature representation while retaining the spatial reconstruction capabilities of U-Net.

---

## Project Structure

```text
vein-vision-ai/
│
├── Dataset/
│
├── preprocessing/
│
├── readme_pics/
│
├── src/
│
├── configuration_stare.txt
├── prepare_datasets_STARE.py
├── run_training.py
├── run_testing.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

The repository separates dataset preparation, preprocessing, model implementation, training, and testing to keep the workflow modular.

---

## Dataset

The project works with images containing visible vascular structures and corresponding segmentation masks.

The dataset pipeline includes preprocessing steps such as:

- Image resizing
- Pixel normalization
- Image preparation
- Mask preparation
- Dataset organization
- Optional augmentation

The repository also includes dataset preparation code and STARE-related configuration.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Harshakiran08/vein-vision-ai.git
cd vein-vision-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Prepare the Dataset

Run the dataset preparation script according to the configured dataset paths:

```bash
python prepare_datasets_STARE.py
```

### Train the Model

```bash
python run_training.py
```

### Test the Model

```bash
python run_testing.py
```

> Dataset paths and configuration values may need to be adjusted for the local environment before running the scripts.

---

## Results

Add your **actual experimental results** here.

For example:

| Model | Dice Score | IoU | Loss |
|---|---:|---:|---:|
| U-Net | `XX.XX%` | `XX.XX%` | `X.XXXX` |
| R2U-Net | `XX.XX%` | `XX.XX%` | `X.XXXX` |

Only publish values produced by your actual experiments.

### Visual Results

Add representative examples showing:

```text
Input Image → Ground Truth → U-Net Prediction → R2U-Net Prediction
```

Recommended images:

- Original input
- Ground-truth segmentation mask
- U-Net output
- R2U-Net output
- Side-by-side comparison

---

## Applications

The techniques explored in this project can be relevant to research areas such as:

- Biomedical image analysis
- Medical image segmentation
- Vascular structure visualization
- Computer-assisted imaging research
- Deep learning research

The project does **not** provide medical diagnosis or clinical recommendations.

---

## Future Improvements

Potential improvements include:

- Attention-based segmentation architectures
- Improved data augmentation
- Hyperparameter optimization
- More comprehensive evaluation metrics
- Real-time image segmentation
- GPU-optimized inference
- Model deployment through an API
- Web-based visualization interface
- Mobile or edge-device experimentation

---

## Learning Outcomes

This project provided practical experience with:

- Deep learning model development
- Semantic image segmentation
- U-Net architecture
- R2U-Net architecture
- Medical image preprocessing
- Dataset preparation
- Model training and evaluation
- Computer vision workflows
- Python-based ML project organization

---

## License

This project is licensed under the **MIT License**.

---

## Author

**Harsha Kiran H B**

B.E. — Information Science & Engineering

GitHub: [@Harshakiran08](https://github.com/Harshakiran08)
