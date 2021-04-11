# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 23:15:35 2021

Pandas

@author: JohanL
"""

import pandas as pd

"""
Series
"""

age = pd.Series([10,20,30,40], index=['age1', 'age2', 'age3', 'age4'])
age.age3
Filtered_age = age[age>20]

# Calling Values of the Series
age.values

# Calling Indices of the Series
age.index

age.index = ['A1', 'A2', 'A3', 'A4']
age.index

"""
DataFrame
"""

import numpy as np

df = np.array([[15, 6, 8], [16, 7, 9], [17, 7, 8], [16, 9, 8]])

Data_Set = pd.DataFrame(df, index=['S1','S2','S3','S4'], columns=['Age', 'Grade1', 'Grade2'])

# Add a column
Data_Set['Grade3'] = [8, 7, 7, 6]

Data_Set.loc['S2']
Data_Set.loc[1][3] # .loc works with labels or boolean values
Data_Set.loc['S2']['Grade3']
Data_Set.iloc[1][3]
Data_Set.iloc[1,3]

Data_Set.iloc[:,3]
Data_Set.iloc[:,0]
filtered_Data = Data_Set.iloc[:,1:3] # it show columns 1 and 2 (col3 is not included)

# Drop a column
Data_Set.drop('Grade2', axis=1)

# replace a specific value
Data_Set.replace(7, 7.5, inplace=True)
Data_Set = Data_Set.replace({7.5:1, 6:3})

# quickly review the dataset
Data_Set.head(3)
Data_Set.tail(2)

# Ordering
Data_Set.sort_values('Grade1', ascending=True)
Data_Set.sort_index(axis=0, ascending=True)






