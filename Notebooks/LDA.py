import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import evaluate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.preprocessing import scale_features
from src.model import train_knn_class
from src.data_loader import get_data_split


def LDA():
    X_train, X_test, y_train, y_test = get_data_split()

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # LDA
    #X_train_lda, X_test_lda, lda = apply_lda(X_train_scaled, X_test_scaled, y_train)
    lda = LinearDiscriminantAnalysis()
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)

    # KNN
    knn = train_knn_class(X_train_lda, y_train, k=5)

    # evaluation
    accuracy, cm, precision__score, recall__score ,f1__score = evaluate(knn, X_test_lda, y_test)

    print("LDA + KNN Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)
    print("precision_score", precision__score)
    print("recall_score", recall__score)
    print("f1_score at", f1__score)




def lda_y_score():

    X_train, X_test, y_train, y_test = get_data_split()
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    lda = LinearDiscriminantAnalysis()
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)

    knn = train_knn_class(X_train_lda, y_train, k=5)
    y_score_lda = knn.predict_proba(X_test_lda)[:, 1]
    return y_score_lda,y_test

if __name__=="__main__":
    LDA()


"""
LDA + KNN Accuracy: 0.8857615894039735
Confusion Matrix:
 [[982  41]
 [ 97  88]]
precision_score 0.6821705426356589
recall_score 0.4756756756756757
f1_score at 0.5605095541401274
 """