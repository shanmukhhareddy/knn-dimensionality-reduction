import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import evaluate
from src.preprocessing import scale_features
from src.model import train_knn_class

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# KNN
knn = train_knn_class(X_train_pca, y_train, k=5)

# evaluation
accuracy, cm = evaluate(knn, X_test_pca, y_test)

print("PCA + KNN Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

