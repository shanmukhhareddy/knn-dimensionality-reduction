import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from PCA import pca_y_score
from LDA import lda_y_score
from RP import y_score

y_score_pca, y_test = pca_y_score()
y_score_lda, y_test = lda_y_score()
y_score_rpg, y_score_rps, y_test= y_score()

fpr_pca, tpr_pca, _ = roc_curve(y_test, y_score_pca)
fpr_lda, tpr_lda, _ = roc_curve(y_test, y_score_lda)
fpr_rpg,  tpr_rpg,  _ = roc_curve(y_test, y_score_rpg)
fpr_rps,  tpr_rps,  _ = roc_curve(y_test, y_score_rps)

auc_pca = auc(fpr_pca, tpr_pca)
auc_lda = auc(fpr_lda, tpr_lda)
auc_rpg  = auc(fpr_rpg,  tpr_rpg)
auc_rps  = auc(fpr_rps,  tpr_rps)

# just assign
auc_rps_, auc_lda_, auc_pca_, auc_rpg_ = auc_rps, auc_lda, auc_pca, auc_rpg
print(auc_rps_, auc_lda_, auc_pca_, auc_rpg_)

fpr_pca_, tpr_pca_, fpr_lda_, tpr_lda_, fpr_rpg_, tpr_rpg_, fpr_rps_, tpr_rps_ = (
    fpr_pca, tpr_pca, fpr_lda, tpr_lda, fpr_rpg, tpr_rpg, fpr_rps, tpr_rps
)
print(fpr_pca_, tpr_pca_, fpr_lda_, tpr_lda_, fpr_rpg_, tpr_rpg_, fpr_rps_, tpr_rps_)



# plot
plt.figure()
plt.plot(fpr_pca, tpr_pca, label=f'PCA + KNN (AUC={auc_pca:.2f})')
plt.plot(fpr_lda, tpr_lda, label=f'LDA + KNN (AUC={auc_lda:.2f})')
plt.plot(fpr_rpg,  tpr_rpg,  label=f'RPG + KNN (AUC={auc_rpg:.2f})')
plt.plot(fpr_rps,  tpr_rps,  label=f'RPS + KNN (AUC={auc_rps:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison of all models")
plt.legend()
plt.grid(True)
plt.show()
plt.close()