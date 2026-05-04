# Food Context Classification using Transfer Learning

## Project Overview

This project studies how pretrained deep learning models perform on a **real-world food context classification task**. Instead of identifying the food item itself, the goal was to classify an image into one of the following categories:

- Home Food  
- Restaurant Food  
- Packaged Food  

The project focuses on understanding how different training strategies such as augmentation and synthetic noise affect classification performance.

---

## Objective

To evaluate whether transfer learning models can distinguish food images based on **scene context**, packaging cues, plating style, and environment.

We also compare multiple data enhancement strategies to study when they help and when they hurt performance.

---

## Dataset

Custom dataset collected from publicly available web images.

### Classes:
- `home_food`
- `restaurant_food`
- `packaged_food`

### Dataset Size:
- ~300+ images total  
- Split into:
  - Train: 70%
  - Validation: 15%
  - Test: 15%

### Real-World Variations:
- Different lighting conditions  
- Background clutter  
- Different camera angles  
- Mixed image quality  

---

## Model Used

Pretrained ResNet-50 (ImageNet weights)

Why ResNet-50:
- Strong image classification baseline
- Efficient transfer learning model
- Good performance on moderate-sized datasets

---

## Experiments Performed

### 1. Baseline Model
No augmentation or synthetic transforms.

### 2. Data Augmentation
Applied:
- Random crop
- Horizontal flip
- Rotation
- Color jitter

### 3. Synthetic Data Robustness
Applied:
- Gaussian blur
- Brightness shift
- Contrast change
- Mild grayscale variation

### 4. Combined Strategy
Used augmentation + synthetic transforms together.

---

## Final Results

| Model | Test Accuracy |
|------|---------------|
| Baseline | **94.52%** |
| Synthetic Data | 93.15% |
| Augmentation | 91.78% |
| Combined | 86.30% |

---

## Key Findings

- Baseline transfer learning achieved the best performance.
- Packaged food was easiest to classify due to strong visual cues.
- Main confusion occurred between Home Food and Restaurant Food.
- Heavy augmentation reduced contextual information.
- Synthetic noise improved robustness but did not beat baseline test accuracy.

---

## Project Structure

```text
food_context_classification/
│── data/
│   ├── raw/
│   ├── train/
│   ├── val/
│   └── test/
│
│── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── train_aug.py
│   ├── evaluate_aug.py
│   ├── train_synth.py
│   ├── evaluate_synth.py
│   ├── train_combo.py
│   └── evaluate_combo.py
│
│── outputs/
│   ├── models/
│   ├── plots/
│   └── confusion_matrices/
│
└── README.md
```

## Installation

### Install dependencies:
```
pip install requirements.txt
```

## How to Run

### Step 1: Prepare Dataset Split
```
python src/prepare_data.py
```
### Step 2: Train Models
#### Baseline
```
python src/train.py
```

#### Augmentation
```
python src/train_aug.py
```
#### Synthetic
```
python src/train_synth.py
```
#### Combined
```
python src/train_combo.py
```
### Step 3: Evaluate Models
#### Baseline
```
python src/evaluate.py
```

#### Augmentation
```
python src/evaluate_aug.py
```
#### Synthetic
```
python src/evaluate_synth.py
```
#### Combined
```
python src/evaluate_combo.py
```

## Outputs Generated

Saved automatically inside outputs/

- Trained model weights (.pth)
- Training curves
- Confusion matrices
- Accuracy metrics

## Conclusion

This project shows that stronger augmentation does not always lead to better results. For context-sensitive classification problems, preserving scene information can be more important than aggressive transformations.

Transfer learning with a pretrained ResNet-50 provided strong performance even with a relatively small custom dataset.