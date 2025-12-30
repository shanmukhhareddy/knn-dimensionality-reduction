import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from PCA import pca_y_score
from LDA import lda_y_score
from RP import y_score

y_score_pca, y_test_pca = pca_y_score()
y_score_lda, y_test_lda = lda_y_score()
y_score_rp, y_test_rp = y_score()

fpr_pca, tpr_pca, _ = roc_curve(y_test_pca, y_score_pca)
fpr_lda, tpr_lda, _ = roc_curve(y_test_lda, y_score_lda)
fpr_rp,  tpr_rp,  _ = roc_curve(y_test_rp, y_score_rp)

auc_pca = auc(fpr_pca, tpr_pca)
auc_lda = auc(fpr_lda, tpr_lda)
auc_rp  = auc(fpr_rp,  tpr_rp)

# plot
plt.figure()
plt.plot(fpr_pca, tpr_pca, label=f'PCA + KNN (AUC={auc_pca:.2f})')
plt.plot(fpr_lda, tpr_lda, label=f'LDA + KNN (AUC={auc_lda:.2f})')
plt.plot(fpr_rp,  tpr_rp,  label=f'RP + KNN (AUC={auc_rp:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()