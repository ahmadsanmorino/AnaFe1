
# Typically, feature selection occurs after data preprocessing. 
# The process of selecting the best feature is known as feature selection. 

# In this template, we combine three mechanisms:
# Extra Tree Classifier (To get Information Gain from each feature)
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

X = data.iloc[:,0:11]  #independent columns
y = data.iloc[:,-1]    #dependent columns

# 1st mechanism: Extra Tree Classifier

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

#END