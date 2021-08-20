# # Analysis for Support Vector Machine Classification

# With this template, users can perform analysis 
# using confusion-matrix, accuracy, precision, sensitivity, f1-score
# and receiver operating characteristic curve in one go easily

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv")

X = dataset.drop('TARGET', axis=1)
y = dataset['TARGET']

# Split dataset: Training set & Testing set (70:30)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

# SVM Classifier
from sklearn.svm import SVC
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

#Display confusion-matrix percentage in heat map
matrix_table = confusion_matrix(y_test,y_pred)
print(matrix_table)

sns.heatmap(matrix_table, annot=True)

sns.heatmap(matrix_table/np.sum(matrix_table), annot=True, 
            fmt='.2%', cmap='Oranges')
plt.show()

# SVM Classification report (Accuracy, Precision, Sensitivity, f1-score) 
print(classification_report(y_test,y_pred))

# Receiver Operating Characteristic Curve (ROC)

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()

y_pred = y_pred[:, 1]

# Display area under the curve score (AUC Score)
auc = roc_auc_score(y_test, y_pred)
print('AUC: %.2f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plot_roc_curve(fpr, tpr)

#END