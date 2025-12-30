
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import evaluate
from src.preprocessing import scale_features
from src.model import train_knn_class
from sklearn.metrics import roc_curve, auc

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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

# KNN (baseline)
values=[3,5,11,15,20]


for k in values:
    knn = train_knn_class(X_train_scaled, y_train, k=k)
    y_scores = knn.predict_proba(X_test_scaled)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'k={k} (AUC={roc_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Different K Values")
plt.legend()
plt.grid(True)
plt.show()