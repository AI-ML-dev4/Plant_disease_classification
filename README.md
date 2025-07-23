# 🍎 Apple Leaf Disease Classification

This project is part of my Celebal Summer Internship, where we aim to classify apple leaf diseases using traditional machine learning techniques. The goal is to automatically identify common diseases affecting apple plants by analyzing leaf images.

---

## 📌 Problem Statement

Early identification of plant diseases can save agricultural yield and minimize pesticide usage. In this project, we classify apple leaves into the following categories:

- **Healthy**
- **Multiple Diseases**
- **Rust**
- **Scab**

---

## 📊 Dataset
- https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data
- The dataset consists of over 3000+ apple leaf images**.
- `train.csv` contains the labeled training set.
- `test.csv`
- Each image is classified into one of the 4 disease types

---

## 🔍 Feature Engineering

We extracted rich image features using the following techniques:

- **HOG (Histogram of Oriented Gradients)** for shape
- **Color Histogram** for color distribution
- **LBP (Local Binary Patterns)** for texture
- **Canny Edge Detection** for boundary information

All features were concatenated into a single vector for each image.

---

## 🤖 Model

- Model Used: `Randomforestclassifier`
- Libraries: `OpenCV`, `scikit-learn`, `matplotlib`, `skimage`, `pandas`, `numpy`

---

## 📈 Results

Here are a few screenshots of the output:

### 🔹 Screenshot
![Sample Image](Screenshot (165).png)
---
