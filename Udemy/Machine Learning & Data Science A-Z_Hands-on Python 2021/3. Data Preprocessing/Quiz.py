# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:20:43 2021

@author: JohanL

Quiz module 3
"""

import numpy as np
import pandas as pd

m = np.array([2,5,12,13,15,32,41])

m_mean = np.mean(m)
m_median = np.median(m)

data = pd.DataFrame(m)
data.describe()

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

mild_Uppb = Q3 + 1.5*IQR

"""
Mild Outlier

Upper Bound = 46
"""