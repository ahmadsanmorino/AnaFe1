
# Typically, feature selection occurs after data preprocessing. 
# The process of selecting the best feature from a set of features

# In this template, we combine three mechanisms:
# Extra Tree (To get Information Gain from each feature)
# Chi-Square (To measure how strong the relationship between features)
# Correlation Coefficient (To shows the correlation between features)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

# Read dataset from csv file
data = pd.read_csv("dataset.csv")

X = data.iloc[:,0:12]  # columns for independent features
y = data.iloc[:,-1]    # column for dependent feature

# 1st mechanism: Extra Tree

gain = ExtraTreesClassifier()
gain.fit(X,y)

# Use inbuilt class feature_importances of Extra-Tree classifier
# Print gain score for each feature
print(gain.feature_importances_) 

# Display gain score in a graph
info_gain = pd.Series(gain.feature_importances_, index=X.columns)
info_gain.nlargest(8).plot(kind='bar')
plt.show()

# 2nd mechanism: Chi-Square

# Use inbuilt class SelectKBest to select the top 8 features
# This top feature has the strongest relationship to target feature
topfeatures = SelectKBest(score_func=chi2, k=8)
fit = topfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

# Display the result in two columns, by concatenating two dataframes
ChiScores = pd.concat([dfcolumns,dfscores],axis=1)

# Labeling the columns
ChiScores.columns = ['Features','Chi-Square Score']

# Display 8 top features
print(ChiScores.nlargest(8,'Chi-Square Score')) 

# 3rd mechanism: Correlation Coefficient

# Show the correlation between features using inbuilt class Correlation-Coefficient
corrcoe = data.corr()
correlation_score = corrcoe.index

# Set the area of the heat map
plt.figure(figsize=(10,8))

# Display correlation score each feature in the heat map
g=sns.heatmap(data[correlation_score].corr(),annot=True,cmap="PiYG")
plt.show()

# Selected features are tested using machine learning classifier

X = dataset.drop('TARGET', axis=1)
y = dataset['TARGET']

# Split dataset: Training set & Testing set (70:30)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

# Decision Tree Classifier
from sklearn import tree
dtc = tree.DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

#Display confusion-matrix percentage in heat map
matrix_table = confusion_matrix(y_test,y_pred)
print(matrix_table)

sns.heatmap(matrix_table, annot=True)

sns.heatmap(matrix_table/np.sum(matrix_table), annot=True, 
            fmt='.2%', cmap='Oranges')
plt.show()

# RF Classification report (Accuracy, Precision, Sensitivity, f1-score) 
print(classification_report(y_test,y_pred))

# Receiver Operating Characteristic Curve (ROC)

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('RF Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()

y_pred = y_pred[:, 1]

# Display area under the curve score (AUC Score)
auc = roc_auc_score(y_test, y_pred)
print('AUC: %.2f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plot_roc_curve(fpr, tpr)

#END