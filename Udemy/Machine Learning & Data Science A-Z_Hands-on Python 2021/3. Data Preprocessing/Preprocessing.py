# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 01:07:44 2021

@author: JohanL

Preprocessing

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Data_set1 = pd.read_csv('Data_Set.csv')

# redefine the header
Data_set2 = pd.read_csv('Data_Set.csv', header=2)

# change a column name
Data_set3 = Data_set2.rename(columns = {'Temperature':'Temp'})

# delete a column
Data_set4 = Data_set3.drop('No. Occupants', axis=1)
# another way to do the same as before
Data_set3.drop('No. Occupants', axis=1, inplace=True)

# drop duplicate row
Data_set5 = Data_set4.drop(2, axis=0)
# reset index
Data_set6 = Data_set5.reset_index(drop=True)



# show statistics about our dataset
Data_set6.describe()

# for example, the negative -4 in col E_Heat has not sense
min_item = Data_set6['E_Heat'].min()

# we need to get the index of this min_item
Data_set6['E_Heat'][Data_set6['E_Heat'] == min_item]

# replace the value
Data_set6['E_Heat'].replace(-4,21, inplace=True)



# Covariance
Data_set6.cov()

# Let's build the graph for this
import seaborn as sn

sn.heatmap(Data_set6.corr())


"""
Missing Values
"""

# show information about the dataset
Data_set6.info()
# but it not shows all the missing values. In the column Price there is an entry with
# the value !. It is also a missing value. Since we want to treat it as missing value we do this:
    
Data_set7 = Data_set6.replace('!', np.NaN)

Data_set7.info()

Data_set7 = Data_set7.apply(pd.to_numeric)
Data_set7.isnull()
# Two options, delete the rows with nan values, or fill the nan entries with some value 
# Delete the rows
Data_set7.dropna(axis=0, inplace=True)

# fill the nan entries with some value (it is better)
# It is a very good method in order to not lose in your data.
Data_set8 = Data_set7.fillna(method='ffill')


# What if we want to replace the nan value using the average, or mean, ?
from sklearn.impute import SimpleImputer
M_Var = SimpleImputer(missing_values=np.nan, strategy='mean')
M_Var.fit(Data_set7)
Data_set9 = M_Var.transform(Data_set7)


""""""""""
Outlier Detection
""""""""""

Data_set8.boxplot()

Data_set8['E_Plug'].quantile(0.25)
Data_set8['E_Plug'].quantile(0.75)

"""
Q1 = 21.25
Q3 = 33.75
IQR = Q3 - Q1 = 33.75 - 21.25 = 12.5

Mild Outlier

lower Bound = Q1 - 1.5*IQR = 2.5
Upper Bound = Q3 + 1.5*IQR = 52.5

Extreme Outlier
Upper Bound = Q3 + 3*IQR = 71.25
"""

Data_set8['E_Plug'].replace(120,42, inplace=True)



"""""""""""
Concatenation

""""""""""


New_col = pd.read_csv('Data_New.csv')

Data_set10 = pd.concat([Data_set8, New_col], axis=1)



"""""""""
Dummy Variables

"""""""""

Data_set10.info()
Data_set11 = pd.get_dummies(Data_set10)
Data_set11.info()


"""""""""""
Normalization

"""""""""""

from sklearn.preprocessing import minmax_scale, normalize

# First method: Min Max Scale

Data_set12 = minmax_scale(Data_set11, feature_range=(0,1))

# axis=0 is for normalizing features / axis=1 is for normalizing each sample
Data_set13 = normalize(Data_set11, norm='l2', axis=0) # but it change the Type

Data_set13 = pd.DataFrame(Data_set13, columns=['Time', 'E_Plug', 'E_Heat', 'Price', 'Temp', 'OffPeak', 'Peak'])














































































