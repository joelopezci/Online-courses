# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:39:22 2021

@author: JohanL
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


from sklearn.datasets import load_boston
house = load_boston()

""""""" Preprocessing section """""""

# house.data
Boston_p = pd.DataFrame(house.data, columns=house.feature_names)

# Complete dataset
Boston_p['Target'] = house.target


# Boston_p.describe()
# Boston_p.info()

# Covariance matrix
Boston_p.cov() #It is hard to check the matrix without a graph
# Data_h.corr()
sn.heatmap(Boston_p.corr())

x = Boston_p.iloc[:,0:13]
y = Boston_p.iloc[:,13]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                    train_size=0.75,
                                                    random_state=88)

# Normalizing the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# It is necessary to reshape the y_train set
y_train = np.array(y_train)
y_train = y_train.reshape(-1, 1)
y_train = sc.fit_transform(y_train)


""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""
Multiple Linear Regression

"""""""""

from sklearn.linear_model import LinearRegression

Linear_R = LinearRegression()
Linear_R.fit(X_train, y_train)

Predicted_values_MLR = Linear_R.predict(X_test)
Predicted_values_MLR = sc.inverse_transform(Predicted_values_MLR)



"""""""""
Evaluation Metrics

"""""""""

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math

MAe = mean_absolute_error(y_test, Predicted_values_MLR)

MSe = mean_squared_error(y_test, Predicted_values_MLR)

RMSe = math.sqrt(MSe)

R2 = r2_score(y_test, Predicted_values_MLR)

# defining the error metric by hand
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true-y_pred)/y_true)*100

MAPe = mean_absolute_percentage_error(y_test, Predicted_values_MLR)


"""""""""
Polynomial Linear Regression

"""""""""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


from sklearn.datasets import load_boston
Boston_p = load_boston()

x = Boston_p.data[:,5] #number of rooms
y = Boston_p.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                    train_size=0.75,
                                                    random_state=88)
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)

from sklearn.preprocessing import PolynomialFeatures
Poly_p = PolynomialFeatures(degree=2)

Poly_X_train = Poly_p.fit_transform(X_train)
Poly_X_test = Poly_p.fit_transform(X_test)

from sklearn.linear_model import LinearRegression
Linear_R = LinearRegression()
Poly_LR = Linear_R.fit(Poly_X_train, y_train)

Predicted_value_p = Poly_LR.predict(Poly_X_test)

# Evaluating the model
from sklearn.metrics import r2_score
R2 = r2_score(y_test, Predicted_value_p)

# We can choose others matrics...homework



"""""""""
Random Forest

"""""""""

from sklearn.ensemble import RandomForestRegressor
Random_F = RandomForestRegressor(n_estimators=500, max_depth=20 , random_state=33)
Random_F.fit(X_train, y_train)

Predicted_values_RF = Random_F.predict(X_test)
Predicted_values_RF = Predicted_values_RF.reshape(-1,1)
Predicted_values_RF = sc.inverse_transform(Predicted_values_RF)


# Evaluation Metrics

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math

MAe = mean_absolute_error(y_test, Predicted_values_RF)

MSe = mean_squared_error(y_test, Predicted_values_RF)

RMSe = math.sqrt(MSe)

R2 = r2_score(y_test, Predicted_values_RF)

# defining the error metric by hand
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true-y_pred)/y_true)*100

MAPe = mean_absolute_percentage_error(y_test, Predicted_values_RF)


"""""""""
SVR

"""""""""

from sklearn.svm import SVR

Regressor_SVR = SVR(kernel='rbf')

Regressor_SVR.fit(X_train, y_train)

Predicted_values_SVR = Regressor_SVR.predict(X_test)
Predicted_values_SVR = Predicted_values_SVR.reshape(-1,1)
Predicted_values_SVR = sc.inverse_transform(Predicted_values_SVR)

# Evaluation Metrics

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math

MAe = mean_absolute_error(y_test, Predicted_values_SVR)

MSe = mean_squared_error(y_test, Predicted_values_SVR)

RMSe = math.sqrt(MSe)

R2 = r2_score(y_test, Predicted_values_SVR)

# defining the error metric by hand
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true-y_pred)/y_true)*100

MAPe = mean_absolute_percentage_error(y_test, Predicted_values_SVR)

# Homework: Compare all the regression models

















