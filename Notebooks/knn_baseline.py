import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import evaluate
from src.preprocessing import scale_features
from src.model import train_knn_class
from src.data_loader import get_data_split

import pandas as pd
import matplotlib.pyplot as plt


# train-test split
X_train, X_test, y_train, y_test = get_data_split()
# scaling
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

# KNN (baseline)
values=[3,5,7,11,15]

accuracies = []
precision = []
recall=[]
f1_score = []

for i in values:

    knn = train_knn_class(X_train_scaled, y_train, k=i)

    # evaluation
    accuracy, cm, precision_knn, recall_knn, f1 = evaluate(knn, X_test_scaled, y_test)

    accuracies.append(accuracy)
    precision.append(precision_knn)
    recall.append(recall_knn)
    f1_score.append(f1)


    print("Metrices at k=",i)
    print("Baseline KNN Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)
    print("Precision:", precision_knn)
    print("recall:",recall_knn)
    print("F1-score:", f1)
    print()


plt.figure()
plt.plot(values, accuracies , marker='o', label='accuracy')
plt.plot(values, precision , marker='o', label='precision')
plt.plot(values, recall , marker='o', label='Recall')
plt.plot(values, f1_score, marker='o', label='F1-score')
plt.xlabel("Number of Dimensions")
plt.ylabel("metrices")
plt.title("Accuracy vs Number of Dimensions")
plt.legend()
plt.show()
plt.close()


'''
Metrices at k= 3
Baseline KNN Accuracy: 0.8485099337748344
Confusion Matrix:
 [[975  48]
 [135  50]]
Precision: 0.5102040816326531
recall: 0.2702702702702703
F1-score: 0.35335689045936397

Metrices at k= 5
Baseline KNN Accuracy: 0.847682119205298
Confusion Matrix:
 [[986  37]
 [147  38]]
Precision: 0.5066666666666667
recall: 0.20540540540540542
F1-score: 0.2923076923076923

Metrices at k= 7
Baseline KNN Accuracy: 0.8526490066225165
Confusion Matrix:
 [[996  27]
 [151  34]]
Precision: 0.5573770491803278
recall: 0.1837837837837838
F1-score: 0.2764227642276423

Metrices at k= 11
Baseline KNN Accuracy: 0.859271523178808
Confusion Matrix:
 [[1014    9]
 [ 161   24]]
Precision: 0.7272727272727273
recall: 0.12972972972972974
F1-score: 0.22018348623853212

Metrices at k= 15
Baseline KNN Accuracy: 0.8584437086092715
Confusion Matrix:
 [[1015    8]
 [ 163   22]]
Precision: 0.7333333333333333
recall: 0.11891891891891893
F1-score: 0.20465116279069767
'''