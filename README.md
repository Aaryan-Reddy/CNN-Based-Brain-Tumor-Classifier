# CNN-Based-Brain-Tumor-Classifier

Classify brain MRI images into four classes — glioma, meningioma, pituitary, and no tumor — using a TensorFlow/Keras CNN.

---

## Description
This project implements a Convolutional Neural Network (CNN) to classify MRI images of human brains into four categories: **glioma**, **meningioma**, **pituitary**, or **no tumor**, aiming for an accurate, deployment-ready workflow for preliminary image analysis.

---

## Dataset
- **Source:** [masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) on Kaggle (CC0-1.0).  
- **Total Samples:** 7,023 images initially found  
- **Training Data:** 6,380 images after oversampling for balance  
- **Test Data:** 1,311 images  

---

## Model Architecture
Sequential CNN:  
`Conv2D(32, 3×3, ReLU) → MaxPool(2×2) → Conv2D(64, 3×3, ReLU) → MaxPool(2×2) → Flatten → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) → Dense(4, Softmax)`

| Layer Type   | Key Parameters           | Output Shape    |
|---------------|--------------------------|-----------------|
| Conv2D        | 32 filters, 3×3, ReLU    | (126, 126, 32)  |
| MaxPooling2D  | 2×2                      | (63, 63, 32)    |
| Conv2D        | 64 filters, 3×3, ReLU    | (61, 61, 64)    |
| MaxPooling2D  | 2×2                      | (30, 30, 64)    |
| Flatten       | —                        | 57,600          |
| Dropout       | 0.3                      | 57,600          |
| Dense         | 128 units, ReLU          | 128             |
| Dropout       | 0.2                      | 128             |
| Dense         | 4 units, Softmax         | 4               |

---

## Results and Performance
**Training Configuration:**  
10 epochs, Adam optimizer, Sparse Categorical Cross-Entropy loss.

- **Final Test Accuracy:** 83.40%  
- **Final Test Loss:** 0.485  

**Class-Specific F1-Scores:**
- Glioma: 0.83  
- Meningioma: 0.67  
- No Tumor: 0.88  
- Pituitary: 0.93  

---

## Optimization for Deployment
The trained Keras model was exported and quantized to TensorFlow Lite for efficient edge or mobile deployment.

| Model Type | Size | Reduction |
|-------------|------|-----------|
| Keras Model | 84.64 MB | — |
| Quantized TFLite | 7.06 MB | 91.66% |

---

## Setup and Usage

### Clone
```bash
git clone https://github.com/Aaryan-Reddy/CNN-Based-Brain-Tumor-Classifier.git
cd CNN-Based-Brain-Tumor-Classifier

pip install tensorflow pandas numpy opencv-python scikit-learn matplotlib seaborn
