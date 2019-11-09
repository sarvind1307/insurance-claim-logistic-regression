import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

insuranceDF = pd.read_csv('insurance2.csv')
insuranceDF.head()

insuranceDF.info()

corr = insuranceDF.corr();
print(corr);
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)
plt.show()

dfTrain = insuranceDF[:]
trainLabel = np.asarray(dfTrain['insuranceclaim'])
trainData = np.asarray(dfTrain.drop('insuranceclaim',1))

means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)
trainData = (trainData - means) / stds

insuranceCheck = LogisticRegression(solver="newton-cg")
insuranceCheck.fit(trainData, trainLabel)
y_pred=insuranceCheck.predict(trainData)

coeff = list(insuranceCheck.coef_[0])
labels = list(dfTrain.drop('insuranceclaim',1).columns)
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 5),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')
plt.show()

accuracy = insuranceCheck.score(trainData, trainLabel)
print("accuracy = ", accuracy * 100, "%")
mean_absolute_error(trainLabel,y_pred)
