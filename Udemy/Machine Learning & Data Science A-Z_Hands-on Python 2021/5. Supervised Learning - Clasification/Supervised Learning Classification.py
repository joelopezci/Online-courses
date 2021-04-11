# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:56:37 2021

@author: JohanL

Supervised Learning Classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()
iris.feature_names

# Make out dataset
Data_iris = iris.data
Data_iris = pd.DataFrame(Data_iris, columns = iris.feature_names)

Data_iris['label'] = iris.target

plt.scatter(Data_iris.iloc[:,2], Data_iris.iloc[:,3], c = iris.target)
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.legend()
plt.show()

x = Data_iris.iloc[:,0:4]
y = Data_iris.iloc[:,4]


"""""""""
K-NN Classifier

"""""""""

from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=1)

kNN.fit(x, y)

x_new1 = np.array([[5.6, 3.4, 1.4, 0.1]])
kNN.predict(x_new1)

x_new2 = np.array([[6.1, 3.3, 3.5, 1.7]])
kNN.predict(x_new2)


########### How to test the model?
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, train_size=0.8,
                                                    random_state=41, shuffle=True,
                                                    stratify=y)
# stratify means that the test set is chosen by taking 20% of each label type randomly

from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=1)

kNN.fit(x_train, y_train)

y_prediction = kNN.predict(x_test)

########### How good or bad is the model?
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_prediction)


"""""""""
Decision Tree

"""""""""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

Dt = DecisionTreeClassifier()

Dt.fit(x_train, y_train)

y_prediction_Dt = Dt.predict(x_test)

accuracy_score(y_test, y_prediction_Dt)


# Decision Tree - Cross Validation

from sklearn.model_selection import cross_val_score

score_Dt = cross_val_score(Dt, x, y, cv=10)



"""""""""
Naive Bayes Classifier

"""""""""

from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()

NB.fit(x_train, y_train)

y_prediction_NB = NB.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_prediction_NB)

# Naive Bayes - Cross Validation

from sklearn.model_selection import cross_val_score

score_NB = cross_val_score(NB, x, y, cv=10)


"""""""""
Logistic Regression

"""""""""

from sklearn.datasets import load_breast_cancer

Data_C = load_breast_cancer()

x = Data_C.data
y = Data_C.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    train_size=0.8,
                                                    random_state=45,
                                                    shuffle=True,
                                                    stratify=y)

from sklearn.linear_model import LogisticRegression
Lr = LogisticRegression()
Lr.fit(x_train, y_train)
predicted_classes_Lr = Lr.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predicted_classes_Lr)

# Cross Validation
from sklearn.model_selection import cross_val_score
score_Lr = cross_val_score(Lr, x, y, cv=10)


"""""""""
Evaluation Metrics

"""""""""

# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report

conf_Mat = confusion_matrix(y_test, predicted_classes_Lr)
class_rep = classification_report(y_test, predicted_classes_Lr) #Very nice report

# ROC Curve
from sklearn.metrics import roc_curve

y_prob = Lr.predict_proba(x_test)
y_prob = y_prob[:,1]

FPR, TPR, Thresholds = roc_curve(y_test, y_prob)

plt.plot(FPR, TPR)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# AUC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_prob)
# AUC = 0.9914021164021164 .....It is a very good model!!

"""""""""""
# Homework: do the same for k-NN and Decision Tree for the Cancer dataset 
# and compare AUC 

"""""""""""

# Let's do it

"""
k-NN Classifier for Cancer dataset

"""
from sklearn.neighbors import KNeighborsClassifier
kNN = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=1)
kNN.fit(x_train, y_train)
predicted_class_kNN = kNN.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predicted_class_kNN)

# Cross Validation
from sklearn.model_selection import cross_val_score
score_kNN = cross_val_score(kNN, x, y, cv=20)


# Evaluation Metrics

# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
conf_Mat_kNN = confusion_matrix(y_test, predicted_class_kNN)
class_rep_kNN = classification_report(y_test, predicted_class_kNN)

# ROC Curve
from sklearn.metrics import roc_curve

y_prob = kNN.predict_proba(x_test)
y_prob = y_prob[:,1]

FPR, TPR, Thresholds = roc_curve(y_test, y_prob)

plt.plot(FPR, TPR)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.show()

# AUC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_prob)
# AUC = 0.964781746031746....Nice model!


"""
Decision Tree for Cancer dataset

"""



























