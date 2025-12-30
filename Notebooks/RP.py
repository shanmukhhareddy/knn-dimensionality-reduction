import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import evaluate
from src.preprocessing import scale_features
from src.model import train_knn_class

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
import matplotlib.pyplot as plt
# load data
df = pd.read_csv("data/processed_data.csv")

X = df.drop('Status', axis=1)
y = df['Status']

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# scaling
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)


dimensions = [5, 10, 15, 20]
gaussian_accuracies = []
gaussian_precision = []
gaussian_recall=[]
gaussian_f1_score = []

sparse_accuracies = []
sparse_precision = []
sparse_recall=[]
sparse_f1_score = []

for d in dimensions:

    # Random Projection models
    gaussian = GaussianRandomProjection(n_components=d, random_state=42)
    sparse   = SparseRandomProjection(n_components=d, random_state=5)

    # Gaussian RP
    X_train_gaussian = gaussian.fit_transform(X_train_scaled)
    X_test_gaussian  = gaussian.transform(X_test_scaled)

    model_gaussian = train_knn_class(X_train_gaussian, y_train, k=5)

    accuracy, cm, precision,recall, f1 = evaluate(model_gaussian, X_test_gaussian, y_test)

    gaussian_accuracies.append(accuracy)
    gaussian_precision.append(precision)
    gaussian_recall.append(recall)
    gaussian_f1_score.append(f1)

    print("Gaussian RP + KNN Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)
    print("Precision:", precision)
    print("recall:",recall)
    print("F1-score:", f1)

    # Sparse RP
    X_train_sparse = sparse.fit_transform(X_train_scaled)
    X_test_sparse  = sparse.transform(X_test_scaled)

    model_sparse = train_knn_class(X_train_sparse, y_train, k=5)

    accuracy, cm, precision,recall, f1 = evaluate(model_sparse, X_test_sparse, y_test)

    sparse_accuracies.append(accuracy)
    sparse_precision.append(precision)
    sparse_recall.append(recall)
    sparse_f1_score.append(f1)

    print("\nSparse RP + KNN Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)
    print("Precision:", precision)
    print("recall:" ,recall)
    print("F1-score:", f1)


plt.figure()
plt.plot(dimensions, gaussian_accuracies, marker='o', label='RPgaussian + KNN')
plt.plot(dimensions, sparse_accuracies, marker='o', label='RPSparse + KNN')
plt.xlabel("Number of Dimensions")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of Dimensions")
plt.legend()
plt.show()

def y_score():
    y_score_rpg = model_gaussian.predict_proba(X_test_gaussian)[:, 1]
    y_score_rps = model_sparse.predict_proba(X_test_sparse)[:, 1]
    return y_score_rpg,y_test