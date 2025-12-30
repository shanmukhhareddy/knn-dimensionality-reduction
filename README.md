# Optimizing KNN Performance with Dimensionality Reduction

## Project Overview
This project explores how dimensionality reduction techniques can improve the performance of the **K-Nearest Neighbors (KNN)** algorithm.  
Since KNN is a distance-based model, its performance often degrades in high-dimensional spaces due to the **curse of dimensionality**.

We compare:
- Baseline KNN
- PCA + KNN
- LDA + KNN  

to analyze accuracy and class-wise performance.

---

## Dataset Description
- Dataset: Breast Cancer Dataset (Kaggle)
- Total samples: **4024**
- Target variable: **Status** (binary classification)

### Raw Features
- Numerical:  
  Age, Tumor Size, Regional Node Examined, Reginol Node Positive, Survival Months
- Categorical:  
  Race, Marital Status, T Stage, N Stage, 6th Stage, Grade, A Stage, Estrogen Status, Progesterone Status

---

## Data Preprocessing
- No missing values detected
- Categorical features handled using:
  - **Label Encoding** for binary categories
  - **One-Hot Encoding** for multi-class categories
- Target variable encoded and moved to the last column
- Final processed dataset:
  - **37 numeric features**
  - Ready for distance-based models like KNN

---

## Project Structure

│
├── data/
│ └── processed_data.csv
│
├── notebooks/
│ ├── knn_baseline.py
│ ├── pca_knn.py
│ └── LDA.py
│
├── src/
│ ├── init.py
│ ├── preprocessing.py
│ ├── dimensionality.py
│ ├── model.py
│ └── evaluation.py
│
└── README.md


---

## Modeling & Results

### 1 Baseline KNN
- K = 5
- Feature Scaling: MinMaxScaler

**Accuracy**


0.8587991718426501


**Confusion Matrix**


[[2017 37]
[ 304 57]]


**Observation**
- Strong performance on majority class
- Weak recall on minority class

---

### 2️ PCA + KNN
- PCA retained **95% variance**
- KNN trained on PCA-transformed features

**Accuracy**


0.8488612836438924


**Confusion Matrix**


[[2004 50]
[ 315 46]]


**Observation**
- Dimensionality reduced
- Slight drop in accuracy
- Minority class performance did not improve

---

### 3️ LDA + KNN
- LDA used as supervised dimensionality reduction
- Class labels used during feature transformation

**Accuracy**


0.8857615894039735


**Confusion Matrix**


[[982 41]
[ 97 88]]


**Observation**
- Best overall accuracy
- Improved class separation
- Better balance between classes

---

## Model Comparison

| Model        | Accuracy |
|-------------|----------|
| Baseline KNN | 0.8588 |
| PCA + KNN    | 0.8489 |
| LDA + KNN    | **0.8858** |

---

## Key Learnings
- KNN is highly sensitive to feature scale and dimensionality
- PCA helps reduce dimensionality but may lose class-discriminative information
- LDA performs best by using class labels during dimensionality reduction
- Supervised dimensionality reduction is more effective for KNN in this dataset

---

## Conclusion
This project demonstrates that KNN performance can be optimized using dimensionality reduction techniques.  
While PCA reduces feature space, **LDA + KNN achieves the best results** by improving class separation and handling high-dimensional data more effectively.

---

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- VS Code

