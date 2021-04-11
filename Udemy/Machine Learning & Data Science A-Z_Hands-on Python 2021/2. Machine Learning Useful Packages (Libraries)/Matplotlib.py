# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 17:14:53 2021

@author: JohanL

Visualization with Matplotlib
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# example No.1
Year = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
Temp_index = [0.72, 0.61, 0.65, 0.68, 0.75, 0.90, 1.02, 0.93, 0.85, 0.99, 1.02]

plt.plot(Year, Temp_index)
plt.xlabel('Year')
plt.ylabel('Temp_Index')
plt.title('Global Temperature', {'fontsize':20, 'horizontalalignment':'center'})
plt.show()

# example No.2
Month = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

Customer1 = [12,13,9,8,7,8,8,7,6,5,8,10]
Customer2 = [14,16,11,7,6,6,7,6,5,8,9,12]

plt.plot(Month, Customer1) # simple graph but we need to give some information.
plt.plot(Month, Customer2)
plt.show()

plt.plot(Month, Customer1, color='red', label='Customer1', marker='o') # complete graph
plt.plot(Month, Customer2, color='blue', label='Customer2', marker='^')
plt.xlabel('Month')
plt.ylabel('Electricity Consumption')
plt.title('Building Consumption')
plt.legend()
# plt.legend(loc='upper left')
plt.show()

# Let's use subplots
plt.subplot(1,2,1)
plt.plot(Month, Customer1, color='red', label='Customer1', marker='o') # complete graph
plt.xlabel('Month')
plt.ylabel('Electricity Consumption')
plt.title('Building Consumption Customer1')
plt.show()

plt.subplot(1,2,2)
plt.plot(Month, Customer2, color='blue', label='Customer1', marker='^') # complete graph
plt.xlabel('Month')
plt.title('Building Consumption Customer2')
plt.show()

# example No.3 scatter plot
plt.scatter(Month, Customer1, label='Customer1', color='red')
plt.scatter(Month, Customer2, label='Customer2', color='blue')
plt.xlabel('Month')
plt.ylabel('Electricity Consumption')
plt.title('Scater Plot of Building Consumption')
plt.grid()
plt.legend()
plt.show()

# example No.4 Histrogram (it show the number of ocurrences of each entry of Customer1: The frecuency)
plt.hist(Customer1, bins=20, color='green')
plt.ylabel('Electricity Consumption')
plt.title('Histogram')
plt.show()

# example No.5 Bar chart
plt.bar(Month, Customer1, width=0.8, color='b')
plt.show()

# example No.6 Bar Chart for two customers
plt.bar(Month, Customer1, color='red', label='customer1')
plt.bar(Month, Customer2, color='blue', label='customer2')
plt.xlabel('Month')
plt.ylabel('Electricity Consumption')
plt.title('Bar Chart')
plt.legend()
plt.show() # but we cannot see the amount of consumption of customer1


bar_width = 0.4
Month_b = np.arange(1,13)
plt.bar(Month_b, Customer1, bar_width, color='red', label='customer1')
plt.bar(Month_b+bar_width, Customer2, bar_width, color='blue', label='curtomer2') # Here is the trick
plt.xlabel('Month')
plt.ylabel('Electricity Consumption')
plt.title('Bar Chart')
plt.legend()
plt.show()


plt.bar(Month_b, Customer1, bar_width, color='red', label='customer1')
plt.bar(Month_b+bar_width, Customer2, bar_width, color='blue', label='curtomer2') # Here is the trick
plt.xlabel('Month')
plt.ylabel('Electricity Consumption')
plt.title('Bar Chart')
# very interesting trick
plt.xticks(Month_b + (bar_width)/12, ('Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
plt.legend()
plt.show()

# example No.7 Boxplot
plt.boxplot(Customer1, vert=False, notch=True) #simple graph

plt.boxplot([Customer1, Customer2], patch_artist=True,
            boxprops=dict(facecolor='skyblue', color='red'),
            whiskerprops=dict(color='blue'),
            capprops=dict(color='orange'),
            medianprops=dict(color='green'))
plt.show()
            













