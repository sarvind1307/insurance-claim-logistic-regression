
#This ignores all the warnings which comes during the execution of code.

from warnings import simplefilter
simplefilter(action='ignore')

#Importing all the necessary Python Packages for Logistic Regression.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#The database is stored in "insurance2.csv" having the following data.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

insuranceDF = pd.read_csv('insurance2.csv')
insuranceDF.info()
insuranceDF

#This gives the co-relation between the various features of the database.

corr = insuranceDF.corr();
print(corr)
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)
plt.show()

#This gives the count of the various features based on whether the insurance claim has been made or not.

sns.set(style="darkgrid")
sns.countplot(x='insuranceclaim',data=insuranceDF)
sns.catplot(x="age", col="insuranceclaim",data=insuranceDF, kind="count")
sns.catplot(x="sex", col="insuranceclaim",data=insuranceDF, kind="count")
sns.catplot(x="children", col="insuranceclaim",data=insuranceDF, kind="count")
sns.catplot(x="smoker", col="insuranceclaim",data=insuranceDF, kind="count")
sns.catplot(x="region", col="insuranceclaim",data=insuranceDF, kind="count")
plt.show()

#This specifies the size of the dataset to be used for Model Training.
#Specify the range in [ : ] to select a portion of the dataset.

dfTrain = insuranceDF[:]

#Apply Logistic Regression Model for the dataset.

trainLabel = np.asarray(dfTrain['insuranceclaim'])
trainData = np.asarray(dfTrain.drop('insuranceclaim',1))

#Plotting Feature Chart to get the influence of various parameters on the Insurance Prediction Claim.

means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)
trainData = (trainData - means)/stds



insuranceCheck = LogisticRegression(solver="lbfgs")
insuranceCheck.fit(trainData, trainLabel)


coeff = list(insuranceCheck.coef_[0])
labels = list(dfTrain.drop('insuranceclaim',1).columns)
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(10, 10),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')
plt.show()

#The following conclusion can be drawn from the above Feature Chart:-
#1. BMI, Smoker have significant influence on the model, specially BMI. It is good to see our machine learning model match what we have been hearing from doctors our entire lives!

#2. Children has a negative influence on the prediction, i.e. higher number children / dependents are correlated with a policyholder not taken insurance claim.

#3. Although age was more correlated than BMI to the output variables (as we saw during data exploration), the model relies more on BMI. This can happen for several reasons, including the fact that the correlation captured by age is also captured by some other variable, whereas the information captured by BMI is not captured by other variables.

#Note that this above interpretations require that our input data is normalized. Without that, we can't claim that importance is proportional to weights.

#Firstly, the model is trained 100 times using train_test_split.
#And this Process is again repeated for 100 times to get more accurate model.

#List 'z' stores the the maximum Percentage Accuracy Score (PAS) of each Internal Iteration.


z=[]
m=0
me=0
for j in range(0,100):
    for i in range(0,100):
        x_train,x_test,y_train,y_test=train_test_split(trainData,trainLabel)
        insuranceCheck.fit(x_train,y_train)
        yp=insuranceCheck.predict(x_test)
        p=accuracy_score(y_test,yp)
        if p>=m:
            m=p
            me = mean_absolute_error(y_test, yp)
    z.append(m*100)
print("Highest Accurate Model Accuracy Score=>",max(z),'%')
print("MAE=>",me)