import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import evaluate
from src.preprocessing import scale_features
from src.model import train_knn_class
from src.data_loader import get_data_split


from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
import matplotlib.pyplot as plt



def RandomProjection():
    X_train, X_test, y_train, y_test = get_data_split()

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

        print("Gaussian RP + KNN Accuracy at", d,":", accuracy)
        print("Confusion Matrix at", d,":", cm)
        print("Precision at", d,":", precision)
        print("recall at", d,":",recall)
        print("F1-score at", d,":", f1)

        # Sparse RP
        X_train_sparse = sparse.fit_transform(X_train_scaled)
        X_test_sparse  = sparse.transform(X_test_scaled)

        model_sparse = train_knn_class(X_train_sparse, y_train, k=5)

        accuracy, cm, precision,recall, f1 = evaluate(model_sparse, X_test_sparse, y_test)

        sparse_accuracies.append(accuracy)
        sparse_precision.append(precision)
        sparse_recall.append(recall)
        sparse_f1_score.append(f1)

        print("\nSparse RP + KNN Accuracy at", d,":", accuracy)
        print("Confusion Matrix at", d,":", cm)
        print("Precision at", d,":", precision)
        print("recall at", d,":",recall)
        print("F1-score at", d,":", f1)


    plt.figure()
    plt.plot(dimensions, gaussian_accuracies, marker='o', label='RPgaussian + KNN')
    plt.plot(dimensions, sparse_accuracies, marker='o', label='RPSparse + KNN')
    plt.xlabel("Number of Dimensions")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Dimensions")
    plt.legend()
    plt.show()

def y_score():

    X_train, X_test, y_train, y_test = get_data_split()
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    gaussian = GaussianRandomProjection(n_components=15, random_state=42)
    sparse   = SparseRandomProjection(n_components=15, random_state=42)

    # Gaussian RP
    X_train_gaussian = gaussian.fit_transform(X_train_scaled)
    X_test_gaussian  = gaussian.transform(X_test_scaled)
    model_gaussian = train_knn_class(X_train_gaussian, y_train, k=5)  

    X_train_sparse = sparse.fit_transform(X_train_scaled)
    X_test_sparse  = sparse.transform(X_test_scaled)
    model_sparse = train_knn_class(X_train_sparse, y_train, k=5)  

    y_score_rpg = model_gaussian.predict_proba(X_test_gaussian)[:, 1]
    y_score_rps = model_sparse.predict_proba(X_test_sparse)[:, 1]
    return y_score_rpg, y_score_rps ,y_test

if __name__=="__main__":
    RandomProjection()


"""
Gaussian RP + KNN Accuracy at 5 : 0.8377483443708609
Confusion Matrix at 5 : [[979  44]
 [152  33]]
Precision at 5 : 0.42857142857142855
recall at 5 : 0.1783783783783784
F1-score at 5 : 0.25190839694656486

Sparse RP + KNN Accuracy at 5 : 0.8344370860927153
Confusion Matrix at 5 : [[975  48]
 [152  33]]
Precision at 5 : 0.4074074074074074
recall at 5 : 0.1783783783783784
F1-score at 5 : 0.24812030075187969
Gaussian RP + KNN Accuracy at 10 : 0.8468543046357616
Confusion Matrix at 10 : [[987  36]
 [149  36]]
Precision at 10 : 0.5
recall at 10 : 0.1945945945945946
F1-score at 10 : 0.2801556420233463

Sparse RP + KNN Accuracy at 10 : 0.8211920529801324
Confusion Matrix at 10 : [[965  58]
 [158  27]]
Precision at 10 : 0.3176470588235294
recall at 10 : 0.14594594594594595
F1-score at 10 : 0.2
Gaussian RP + KNN Accuracy at 15 : 0.8435430463576159
Confusion Matrix at 15 : [[986  37]
 [152  33]]
Precision at 15 : 0.4714285714285714
recall at 15 : 0.1783783783783784
F1-score at 15 : 0.25882352941176473

Sparse RP + KNN Accuracy at 15 : 0.8394039735099338
Confusion Matrix at 15 : [[975  48]
 [146  39]]
Precision at 15 : 0.4482758620689655
recall at 15 : 0.21081081081081082
F1-score at 15 : 0.2867647058823529

Gaussian RP + KNN Accuracy at 20 : 0.8485099337748344
Confusion Matrix at 20 : [[978  45]
 [138  47]]
Precision at 20 : 0.5108695652173914
recall at 20 : 0.25405405405405407
F1-score at 20 : 0.33935018050541516

Sparse RP + KNN Accuracy at 20 : 0.8460264900662252
Confusion Matrix at 20 : [[979  44]
 [142  43]]
Precision at 20 : 0.4942528735632184
recall at 20 : 0.23243243243243245
F1-score at 20 : 0.3161764705882353

    """