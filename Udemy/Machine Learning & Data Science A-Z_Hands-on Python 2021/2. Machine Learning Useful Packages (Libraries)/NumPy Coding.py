# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:07:27 2021

Title: NumPy Coding

@author: JohanL
"""

import numpy as np

NumP_Array = np.array([[1,2,3],[4,7,8]])

np1 = np.array([[1,2],[3,4]])

np2 = np.array([[5,2],[1,7]])

mnp = np1@np2 # matrix product
mnp3 = np.dot(np1, np2)

mnp2 = np1*np2 # product component by component
mnp4 = np.multiply(np1, np2)

sum1 = np1 + np2 # add/subtract element by element
sub1 = np1 - np2
sub2 = np.subtract(np1, np2)

el_sum = np.sum(np1) # add up all the elements inside of the array

# broadcasting example
broad_nump = np1 + 3
np3 = np.array([2,4,5])
broad_nump2 = NumP_Array + np3 # expand np3 until get the same area
D = np.divide([12,14,16],5)

np.math.sqrt(10) # numpy has a math package

# Generate distributions
nd = np.random.standard_normal((3,4)) # normal distribution
ud = np.random.uniform(1,15,(3,4)) # uniform distribution


rn = np.random.rand() # random float No.

random_array = np.random.randint(2,50,(3,4)) # random array of integers

zr = np.zeros((2,5))
ones = np.ones((3,2))


# Example of how to filter in a range
filter_ar = np.logical_and(random_array>20, random_array<35)
f_random_ar = random_array[filter_ar]

# Basic statistics
data_n = np.array([1,3,2,7,8,9,11,4])
mean_n = np.mean(data_n)
median_n = np.median(data_n)
var_n = np.var(data_n)
sd_n = np.std(data_n)

var_NumP = np.var(NumP_Array, axis=0) # by columns
var_NumP2 = np.var(NumP_Array, axis=1) # by rows

