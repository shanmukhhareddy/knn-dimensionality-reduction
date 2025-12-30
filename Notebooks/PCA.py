import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import evaluate
from src.preprocessing import scale_features
from src.model import train_knn_class

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


if __name__ == "__main__":
    y_score, y_test = pca_y_score()

# load data
df = pd.read_csv("data/processed_data.csv")

X = df.drop("Status", axis=1)
y = df["Status"]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# scaling
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

# PCA
dimensions = [5, 10, 15, 20]
pca_accuracies = []
pca_precision = []
pca_recall=[]
pca_f1_score = []

for d in dimensions:
    pca = PCA(n_components=d)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # KNN
    knn = train_knn_class(X_train_pca, y_train, k=5)

    # evaluation
    accuracy, cm, precision_score, recall_score, f1_score = evaluate(knn, X_test_pca, y_test)

    pca_accuracies.append(accuracy)
    pca_precision.append(precision_score)
    pca_recall.append(recall_score)
    pca_f1_score.append(f1_score)

    print("PCA + KNN Accuracy at", d ," :", accuracy)
    print("Confusion Matrix at", d ," :", cm)
    print("precision_score at", d ," :", precision_score)
    print("recall_score at", d ," :", recall_score)
    print("f1_score at", d ," :", f1_score)



"""
it is best at 15

PCA + KNN Accuracy at 15  : 0.8501655629139073
Confusion Matrix at 15  : [[985  38]
 [143  42]]
precision_score at 15  : 0.525
recall_score at 15  : 0.22702702702702704
f1_score at 15  : 0.3169811320754717

"""

plt.figure()
plt.plot(dimensions, pca_accuracies, marker='o', label='PCA + KNN')
plt.xlabel("Number of Dimensions")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of Dimensions")
plt.legend()
plt.show()


def pca_y_score():
    y_score_pca = knn.predict_proba(X_test_pca)[:, 1]
    return y_score_pca, y_test

