import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.data_loader import get_data_split
from src.preprocessing import scale_features
from src.model import train_knn_class
from src.evaluation import evaluate





def run_pca_experiment():
    # train-test split
    X_train, X_test, y_train, y_test = get_data_split()

    # scaling
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    dimensions = [5, 10, 15, 20]

    pca_accuracies = []
    pca_precision = []
    pca_recall = []
    pca_f1_score = []

    for d in dimensions:
        pca = PCA(n_components=d)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        # KNN
        knn = train_knn_class(X_train_pca, y_train, k=5)

        # evaluation
        accuracy, cm, precision, recall, f1 = evaluate(
            knn, X_test_pca, y_test
        )

        pca_accuracies.append(accuracy)
        pca_precision.append(precision)
        pca_recall.append(recall)
        pca_f1_score.append(f1)

        print(f"\nPCA + KNN | Components = {d}")
        print("Accuracy :", accuracy)
        print("Confusion Matrix:\n", cm)
        print("Precision:", precision)
        print("Recall   :", recall)
        print("F1-score :", f1)

    # plot
    plt.figure()
    plt.plot(dimensions, pca_accuracies, marker="o", label="PCA + KNN")
    plt.xlabel("Number of Dimensions")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Dimensions (PCA)")
    plt.legend()
    plt.grid(True)
    plt.show()




def pca_y_score(n_components=15, k=5):
    X_train, X_test, y_train, y_test = get_data_split()
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    knn = train_knn_class(X_train_pca, y_train, k=k)
    y_score = knn.predict_proba(X_test_pca)[:, 1]

    return y_score, y_test




if __name__ == "__main__":
    run_pca_experiment()


"""
PCA + KNN | Components = 5
Accuracy : 0.8369205298013245
Confusion Matrix:
 [[978  45]
 [152  33]]
Precision: 0.4230769230769231
Recall   : 0.1783783783783784
F1-score : 0.2509505703422053

PCA + KNN | Components = 10
Accuracy : 0.8369205298013245
Confusion Matrix:
 [[972  51]
 [146  39]]
Precision: 0.43333333333333335
Recall   : 0.21081081081081082
F1-score : 0.28363636363636363

PCA + KNN | Components = 15
Accuracy : 0.8501655629139073
Confusion Matrix:
 [[985  38]
 [143  42]]
Precision: 0.525
Recall   : 0.22702702702702704
F1-score : 0.3169811320754717

PCA + KNN | Components = 20
Accuracy : 0.8485099337748344
Confusion Matrix:
 [[986  37]
 [146  39]]
Precision: 0.5131578947368421
Recall   : 0.21081081081081082
F1-score : 0.2988505747126437

"""