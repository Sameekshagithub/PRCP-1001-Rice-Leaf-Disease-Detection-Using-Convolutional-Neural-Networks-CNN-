# 🌾 Rice Leaf Disease Detection Using CNN

**A Data Science Capstone Project — PRCP-1001**

An end-to-end deep learning pipeline that classifies rice leaf images into three major disease categories — **Bacterial Leaf Blight**, **Brown Spot**, and **Leaf Smut** — using a custom CNN and two transfer-learning models (VGG16, MobileNetV2), fully implemented in a single Jupyter notebook.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Pipeline](#-project-pipeline)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Data Augmentation](#-data-augmentation)
- [Model Architectures](#-model-architectures)
- [Model Comparison](#-model-comparison)
- [Results](#-results)
- [Challenges & Solutions](#-challenges--solutions)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🔍 Overview

Rice is one of the most widely cultivated staple food crops in the world, and its yield is heavily affected by leaf diseases. Early and accurate identification allows farmers to apply the correct treatment quickly, reducing crop loss. Manual diagnosis is slow, subjective, and requires expert knowledge that isn't always available in the field.

This project builds an **automated image classification system** using Convolutional Neural Networks that can look at a photograph of a rice leaf and predict which disease is present.

**Problem statement tasks covered:**

| Task | Description | Status |
|---|---|---|
| Task 1 | Complete data analysis report | ✅ |
| Task 2 | CNN classification model for 3 diseases | ✅ |
| Task 3 | Data augmentation technique analysis | ✅ |
| Extra | Model comparison report + production recommendation | ✅ |
| Extra | Challenges faced report | ✅ |

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total images | 120 JPG images |
| Classes | 3 (perfectly balanced — 40 images each) |
| Source | DataMites Capstone Project Dataset |
| Format | Nested `.zip` (Google Drive export) |

**Disease classes:**

<table>
<tr>
<td align="center" width="33%">
<img src="assets/class_bacterial_leaf_blight.png" width="100%"><br>
<b>Bacterial Leaf Blight</b><br>
<sub>Caused by <i>Xanthomonas oryzae</i>. Long, pale, water-soaked stripes along the veins.</sub>
</td>
<td align="center" width="33%">
<img src="assets/class_brown_spot.png" width="100%"><br>
<b>Brown Spot</b><br>
<sub>Caused by <i>Bipolaris oryzae</i>. Small, round, brown lesions scattered across the blade.</sub>
</td>
<td align="center" width="33%">
<img src="assets/class_leaf_smut.png" width="100%"><br>
<b>Leaf Smut</b><br>
<sub>Caused by <i>Entyloma oryzae</i>. Tiny, dense, black angular spots on both surfaces.</sub>
</td>
</tr>
</table>

> *Images above are illustrative diagrams representing each disease's distinguishing visual pattern.*

---

## 🔄 Project Pipeline


1. **Extract** the nested dataset archive (two-stage zip extraction)
2. **Organize** raw images into clean, uniform class folders
3. **Analyze** the data (EDA — class balance, resolution, corrupt-file check)
4. **Split** into train / validation / test (70 / 15 / 15)
5. **Augment** training images in real time
6. **Build & train** three models (baseline CNN, VGG16, MobileNetV2)
7. **Compare** all models on the same held-out test set
8. **Select & save** the best model for production

---

## 🔬 Exploratory Data Analysis

Before any modelling, the dataset was analyzed to answer four questions:

- **Class distribution** — perfectly balanced (40 images/class), so no class-weighting is required.
- **Image dimensions** — resolutions vary across files, so every image is resized to `224×224` in the pipeline.
- **Visual inspection** — the three diseases are clearly distinguishable by shape, color, and texture.
- **Data quality** — every image was verified with PIL; no corrupt files were found.

> **Key finding:** With only 120 images total, the dataset is very small by deep-learning standards. This single fact drives the two most important modelling decisions in this project: **heavy data augmentation** and **transfer learning**.

---

## 🌀 Data Augmentation

With only ~84 training images, a CNN can easily memorize the training set instead of learning general disease features. Real-time augmentation solves this by generating realistic, varied versions of each image every epoch.


| Technique | Parameter | Why |
|---|---|---|
| Rescale | `1./255` | Normalizes pixels to `[0,1]` for stable training |
| Rotation | `rotation_range=20` | Leaves are photographed at different angles |
| Width/Height Shift | `0.2` | Leaf isn't always centered in frame |
| Shear | `0.2` | Simulates hand-held camera distortion |
| Zoom | `0.2` | Leaves photographed at different distances |
| Horizontal Flip | `True` | Mirroring preserves the disease label |
| Fill Mode | `nearest` | Avoids black-border artifacts |

`vertical_flip` was **intentionally excluded** — flipping top-to-bottom can distort how disease patterns relate to a leaf's natural growth axis.

---

## 🧠 Model Architectures

### Model 1 — Baseline CNN (trained from scratch)


A benchmark model: 3 convolution blocks with BatchNorm + MaxPooling, followed by Dense layers with Dropout. Trained entirely from a random initialization — used to demonstrate why transfer learning is necessary on such a small dataset.

### Models 2 & 3 — Transfer Learning (VGG16 / MobileNetV2)


Both reuse a frozen, ImageNet-pretrained convolutional base and train only a small classification head (`GlobalAveragePooling2D → Dense → Dropout → Softmax`). This lets the model leverage features learned from millions of images instead of learning everything from ~84 training samples.

- **VGG16** — deeper, higher parameter count, strong accuracy.
- **MobileNetV2** — ~10x fewer parameters, faster inference, built for mobile/edge deployment.

---

## ⚖️ Model Comparison

All three models are evaluated on the **same held-out test set** using accuracy, precision, recall, F1-score, parameter count, and training time.

*Illustrative pattern: a from-scratch CNN typically shows a widening train/validation gap (overfitting), while a frozen transfer-learning model tracks much closer together.*


---

## ✅ Results

| Model | Expected Behaviour | Recommended For |
|---|---|---|
| Baseline CNN | Weakest — overfits on small data | Benchmark only |
| VGG16 (Transfer Learning) | High accuracy, large & slower | When max accuracy matters most |
| **MobileNetV2 (Transfer Learning)** | **High accuracy, small & fast** | ✅ **Recommended for production** |

> **MobileNetV2** offers the best balance of accuracy, speed, and model size — ideal for a real-world tool a farmer could run on a phone in the field.

*(Exact metric values are produced when the notebook is executed end-to-end on the dataset — see [Usage](#-usage).)*

---

## 🚧 Challenges & Solutions

| Challenge | Root Cause | Solution |
|---|---|---|
| Nested, inconsistently named zip archives | Google Drive auto-exports each folder as its own timestamped sub-zip | Two-stage extraction + copy into clean class folders |
| Only 120 images total | Small curated capstone dataset | Data augmentation + transfer learning + dropout + early stopping |
| Inconsistent image resolutions | Field photos taken with different cameras | All images resized to `224×224` |
| Visually similar lesions between classes | Overlapping symptoms in poor lighting | Confusion matrix analysis per model |
| Wrong/misleading predictions | Class names hardcoded instead of read from generator | Dynamic `class_indices` mapping |
| Data leakage risk | Dataset was split twice in an earlier version | Single, one-time 70/15/15 split |
| Limited compute | Training 3 deep models is expensive | Frozen pretrained bases + EarlyStopping + small batch size |

---

## 📁 Project Structure(For VS CODE)

```
rice-leaf-disease-detection/
│
├── PRCP_1001_Rice_Leaf_Disease_using_CNN.ipynb   # Complete notebook (all tasks)
├── README.md                                      # This file
├── assets/                                        # Diagrams used in this README
│   ├── pipeline_flowchart.png
│   ├── cnn_architecture.png
│   ├── transfer_learning_architecture.png
│   ├── augmentation_illustration.png
│   ├── training_curve_pattern.png
│   ├── confusion_matrix_schematic.png
│   ├── model_comparison.png
│   └── class_*.png
├── PRCP-1001-RiceLeaf.zip                         # Raw dataset (place here before running)
└── requirements.txt                               # Python dependencies
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/rice-leaf-disease-detection.git
cd rice-leaf-disease-detection

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
tensorflow>=2.15
numpy
pandas
opencv-python
pillow
matplotlib
seaborn
scikit-learn
split-folders
```

---

## ▶️ Usage

1. Place `PRCP-1001-RiceLeaf.zip` in the project root (or upload it in the notebook's first cell if using Google Colab).
2. Open `PRCP_1001_Rice_Leaf_Disease_using_CNN.ipynb` in Jupyter or Google Colab.
3. Run all cells top to bottom — the notebook will:
   - Extract and organize the dataset
   - Perform EDA
   - Split, augment, and train all three models
   - Compare models and save the best one (`Rice_Leaf_Disease_Best_Model.keras`)
4. Use the final prediction cell to test the saved model on a new rice leaf photo.

---

## 🛠️ Tech Stack

`Python` · `TensorFlow / Keras` · `OpenCV` · `Pillow` · `NumPy` · `Pandas` · `Matplotlib` · `Seaborn` · `scikit-learn` · `split-folders`

---

## 🔮 Future Improvements

- Collect a larger, more diverse real-world dataset (hundreds of images per class, varied lighting/growth stages)
- Deploy the best model (MobileNetV2) as a mobile/web app for field use
- Add Grad-CAM visualizations to explain which leaf regions drive each prediction
- Expand to additional rice diseases beyond the current three classes

---

## 👤 Author

**Sameeksha**
Team Code: `PTID-AIE-JUL-26-11142` · Project Code: `PRCP-1001`

---

<p align="center"><i>Built as part of a Data Science Capstone Project.</i></p>
