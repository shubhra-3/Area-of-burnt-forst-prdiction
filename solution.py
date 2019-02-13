
    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("train.csv")
#colnames=['Id','X','Y','month','day','5','6','7','8','9','10','11','12','13']
testset = pd.read_csv('test.csv')#, names=colnames, header=None)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset['area'] = (1+dataset['area']) # for 0 problem of log
dataset['area'] = np.log(dataset['area'])
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
test = testset.iloc[:, 1:-1].values

# Encoding categorical data
# Encoding the Independent Variable
labelencoder_X = LabelEncoder()
labelencoder_Xt = LabelEncoder()
#this sequence is important, since toarray() wont work if we have categorical features
# so we have weeks, then months in our final X, then rest other columns in sequence they apper in dataset

X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
test[:, 2] = labelencoder_Xt.fit_transform(test[:, 2])
test[:, 3] = labelencoder_Xt.fit_transform(test[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
onehotencoder = OneHotEncoder(categorical_features = [2])
test = onehotencoder.fit_transform(test).toarray()
test = test[:, 1:]

onehotencoder = OneHotEncoder(categorical_features = [12])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
onehotencoder = OneHotEncoder(categorical_features = [12])
test = onehotencoder.fit_transform(test).toarray()
test = test[:, 1:]


"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[: ,1:3])
X[: ,1:3] =imputer.transform(X[: ,1:3])"""

"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[: ,0] = labelencoder_X.fit_transform(X[: ,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y=LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)"""

from sklearn.cross_validation import train_test_split
X_train ,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0, random_state=0)

"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)"""


import statsmodels.formula.api as sm
X = np.append(arr = np.ones((450 ,1)).astype(int), values = X , axis = 1)

X_opt = X[:, [0 ,1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,17,18,19,20,21,22,23,24,25,26]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#26
X_opt = X[:, [0 ,1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,17,18,19,20,21,22,23,24,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#4
X_opt = X[:, [0 ,1 ,2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,17,18,19,20,21,22,23,24,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#1
X_opt = X[:, [0 ,2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,17,18,19,20,21,22,23,24,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#7
X_opt = X[:, [0 ,2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15,16,17,18,19,20,21,22,23,24,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#18
X_opt = X[:, [0 ,2, 3, 5, 6, 8, 9,10, 11, 12, 13, 14, 15,16,17 ,19,20,21,22,23,24,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#10
X_opt = X[:, [0 ,2, 3, 5, 6, 8, 9, 11, 12, 13, 14, 15,16,17 ,19,20,21,22,23,24,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#11
X_opt = X[:, [0 ,2, 3, 5, 6, 8, 9, 12, 13, 14, 15,16,17 ,19,20,21,22,23,24,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#24
X_opt = X[:, [0 ,2, 3, 5, 6, 8, 9, 12, 13, 14, 15,16,17 ,19,20,21,22,23,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#13
X_opt = X[:, [0 ,2, 3, 5, 6, 8, 9, 12, 14, 15,16,17 ,19,20,21,22,23,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#19
X_opt = X[:, [0 ,2, 3, 5, 6, 8, 9, 12, 14, 15,16,17,20,21,22,23,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#12
X_opt = X[:, [0 ,2, 3, 5, 6, 8, 9, 14, 15,16,17,20,21,22,23,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#6
X_opt = X[:, [0 ,2, 3, 5, 8, 9, 14, 15,16,17,20,21,22,23,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#22
X_opt = X[:, [0 ,2, 3, 5, 8, 9, 14, 15,16,17,20,21,23,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#2
X_opt = X[:, [0, 3, 5, 8, 9, 14, 15,16,17,20,21,23,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#3
X_opt = X[:, [0 , 5, 8, 9, 14, 15,16,17,20,21,23,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#0
X_opt = X[:, [5, 8, 9, 14, 15,16,17,20,21,23,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#5
X_opt = X[:, [ 8, 9, 14, 15,16,17,20,21,23,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#14
X_opt = X[:, [ 8, 9, 15,16,17,20,21,23,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#15
X_opt = X[:, [ 8, 9,16,17,20,21,23,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()


#9
X_opt = X[:, [ 8,16,17,20,23,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()

#20
X_opt = X[:, [ 8,16,17,23,25]]
regressor_OLS = sm.OLS( endog = Y ,exog = X_opt).fit()
regressor_OLS.summary()


X_trainopt = X_train[:, [ 8,16,17,23,25]]
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_trainopt , Y_train)

X_testopt = X_test[:, [ 8,16,17,23,25]]
Y_pred = regressor.predict(X_testopt)

from sklearn.metrics import mean_squared_error
from math import sqrt
Y_pred[:, 0] = np.exp(Y_pred[:, 0]) 
rms = sqrt(mean_squared_error(Y_test, Y_pred))

test_opt = test[:, [ 8,16,17,23,25]]
pred = regressor.predict(test_opt)

#pred_anti = [(np.exp(x)) for x in [i for i in pred]]

solution = pd.read_csv('sampleSubmission.csv')
solution = pd.DataFrame({'area':pred, 'Id':testset['Id']})
solution['area']=np.exp(solution['area'])
solution['area']=1+solution['area']
solution.to_csv('sampleSubmission.csv',index = False, sep=',',  header=True, columns=["Id","area"])

"""
from sklearn.tree import DecisionTreeClassifier
regressorDT = DecisionTreeClassifier(random_state = 0)
regressorDT.fit(X_train , Y_train)

Y_predDT = regressorDT.predict(X_test)
rmsDT = sqrt(mean_squared_error(Y_test, Y_predDT))
"""
from sklearn.ensemble import RandomForestClassifier
regressorRF = RandomForestClassifier(n_estimators = 100,random_state = 0)
regressorRF.fit(X_train , Y_train)

Y_predDT = regressorRF.predict(X_test)
rmsRF = sqrt(mean_squared_error(Y_test, Y_predRF))