# 🫁 Chest X-Ray Pneumonia Classification

> A deep learning-based system for automated multi-class classification of chest X-ray images — distinguishing **COVID-19 pneumonia**, **common pneumonia**, and **normal** lungs using state-of-the-art CNN architectures.

---

## 📌 Overview

This project implements and compares multiple Convolutional Neural Network (CNN) architectures for pneumonia detection from chest X-ray images. The work is grounded in a published research paper (*Biomedical Signal Processing and Control, 2026*) and extends it by applying **DenseNet121 with two-stage fine-tuning**, achieving **93.44% test accuracy** on a balanced 3-class dataset.

**Classification Target Classes:**
- `COVID-19` — Pneumonia caused by the COVID-19 virus
- `Normal` — Healthy (non-pathological) lungs
- `Pneumonia` — Common bacterial or viral pneumonia

---

## 🧠 Model Architectures Compared

| Architecture | Parameters | Key Strength | Test Accuracy |
|---|---|---|---|
| **VGG-19** | ~143M | Deep uniform architecture, best feature extraction | **97%** *(paper)* |
| **DenseNet121** | ~8M | Dense connections, excellent gradient flow, fewer params | **93.44%** *(this repo)* |
| ResNet | ~25M | Skip connections, handles very deep networks | 95% |
| InceptionV3 | ~23M | Multi-scale feature capture | 94% |
| AlexNet | ~61M | Pioneering deep CNN architecture | 90% |
| SqueezeNet | ~1.2M | Lightweight fire-module design | 88% |

---

## 🏗️ DenseNet121 Architecture

```
Input (224×224×3)
        │
   DenseNet121 Backbone (ImageNet pretrained)
   ├── Dense Block 1  → conv → BN → ReLU × N
   ├── Dense Block 2  → each layer receives ALL prior feature maps
   ├── Dense Block 3  → encourages feature reuse
   └── Dense Block 4  → rich hierarchical representations
        │
   GlobalAveragePooling2D
        │
   BatchNormalization
        │
   Dense(512, ReLU) → Dropout(0.4)
        │
   Dense(256, ReLU) → Dropout(0.3)
        │
   Dense(3, Softmax)  ← COVID-19 | Normal | Pneumonia
```

**Why DenseNet?**
- Each layer receives feature maps from **all preceding layers** → maximum feature reuse
- Stronger gradient flow → no vanishing gradient problem
- Fewer parameters than VGG/ResNet for comparable performance
- Ideal for limited medical datasets

---

## 🔁 Two-Stage Training Strategy

### Stage 1 — Warm-up (Frozen Base)
```python
base_model.trainable = False
optimizer = Adam(lr=1e-3)
epochs = 10
```
The top classification layers are trained first while the DenseNet backbone is frozen. This prevents destroying the pretrained ImageNet weights.

### Stage 2 — Full Fine-Tuning
```python
# Unfreeze last 100 layers of DenseNet
for layer in base_model.layers[:-100]:
    layer.trainable = False

optimizer = Adam(lr=1e-5)
loss = CategoricalCrossentropy(label_smoothing=0.1)
epochs = 30
```
Fine-tuning at a very low learning rate with **label smoothing** prevents overconfidence and improves generalization to unseen X-rays.

---

## 📊 Results

### Overall Performance
```
Overall Test Accuracy: 93.44%
```

### Per-Class Results

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **COVID-19** | 0.99 | 0.94 | **0.96** | 340 |
| **Normal** | 0.86 | 0.98 | **0.92** | 348 |
| **Pneumonia** | 0.97 | 0.88 | **0.92** | 348 |
| *Macro Avg* | 0.94 | 0.93 | 0.94 | 1036 |

### Confusion Matrix

```
                Predicted
              COVID  Normal  Pneumonia
Actual COVID  [ 320     14      6   ]
       Normal [   4    341      3   ]
   Pneumonia  [   0     41    307   ]
```

Key observations:
- **COVID-19 detection is extremely precise** (0.99) — very few false positives
- **Normal class has the highest recall** (0.98) — rarely misses healthy lungs
- Main confusion: pneumonia cases misclassified as normal (41 cases) — a known challenge in chest X-ray classification

---

## 🗂️ Dataset

**Source:** [COVID19-Pneumonia-Normal Chest X-ray PA Dataset](https://www.kaggle.com/datasets/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset) (Kaggle)

| Split | Size | Classes |
|---|---|---|
| Training | 80% (~5,551 images) | COVID-19, Normal, Pneumonia |
| Validation | 20% (~1,388 images) | COVID-19, Normal, Pneumonia |
| Test | Independent set | 1,036 images |

- Total: **6,939 X-ray images** (balanced across 3 classes)
- Screened by an expert radiologist to remove low-quality scans
- All images resized to **224×224×3**

---

## 🔧 Data Augmentation

To prevent overfitting and improve generalization:

```python
ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
```

Additional techniques referenced in the research:
- Brightness & contrast adjustment
- Kernel-based noise introduction
- Random erasing (patch removal)
- Color space transformations

---

## 🛠️ Tech Stack

| Tool | Version |
|---|---|
| Python | 3.8+ |
| TensorFlow / Keras | 2.x |
| NumPy | latest |
| scikit-learn | latest |
| Matplotlib / Seaborn | latest |

---

## 📁 Repository Structure

```
Chest_X_Rays_Pnemonia_3/
├── Dataset/
│   ├── train/
│   │   ├── covid/
│   │   ├── normal/
│   │   └── pneumonia/
│   └── val/
│       ├── covid/
│       ├── normal/
│       └── pneumonia/
├── DENSENET121/          # DenseNet121 training notebooks & scripts
├── VGG16/                # VGG16 experiments
├── EfficientNetB0/       # EfficientNet experiments
├── Data_Augmentation/    # Augmentation pipeline scripts
├── .idea/
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/venkatsaikondra/Chest_X_Rays_Pnemonia_3.git
cd Chest_X_Rays_Pnemonia_3
```

### 2. Install Dependencies
```bash
pip install tensorflow numpy scikit-learn matplotlib seaborn
```

### 3. Prepare the Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset) and place it under `Dataset/train` and `Dataset/val`.

### 4. Train the Model
Update the paths in the script and run:
```bash
python DENSENET121/densenet_train.py
```

---

## 📖 Reference

> Aljuaid, H., Adlan, H., Alkebsi, B., Alfurhood, B. S., Liotta, A., & Cavallaro, L. (2026).
> *An experimental comparison of deep learning models for pneumonia classification from chest X-ray images.*
> **Biomedical Signal Processing and Control**, 112, 108742.
> https://doi.org/10.1016/j.bspc.2025.108742

---

## 👤 Author

**Venkat Sai Kondra**
GitHub: [@venkatsaikondra](https://github.com/venkatsaikondra)

---

## 📄 License

This project is open-source and available for research and educational purposes.
