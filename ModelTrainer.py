import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


'''INPUT PARSED DATA FOR MODEL TRAINING'''
x_test = np.load("parsed_data/x_test.npy")
y_test = np.load("parsed_data/y_test.npy")
x = np.load("parsed_data/x_train.npy")
y = np.load("parsed_data/y_train.npy")


'''Using SGDClassifier'''
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x, y)
#sgd_clf.predict([some_digit])


'''Cross validation score'''
cvs = cross_val_score(sgd_clf, x, y, cv=3, scoring="accuracy")
print("Cross validation score of Stochastic GD over 3 K-Folds")
print(cvs)
print("")


'''Confusion matrix'''
y_train_pred = cross_val_predict(sgd_clf, x, y, cv=3)
conf_m = confusion_matrix(y, y_train_pred)
print("Confusion matrix of Stochastic GD over 3 K-Folds")
print(conf_m)
print("")

y_scores = cross_val_predict(sgd_clf, x, y, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend()
    plt.show()
    
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls[:-1], precisions[:-1], "b-", label="Precision")
    plt.xlabel("Recalls")
    plt.ylabel("Precision")
    plt.show()
    
fpr, tpr, thresholds = roc_curve(y, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.grid(1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC AUC score = "+ str(round(roc_auc_score(y, y_scores), 5)))
    plt.show()
