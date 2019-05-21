'''
This module  will estimate the value of pi 
using a monte carlo method. We compare the 
number of counts that fall between 2 regions 
in a unit square. Those points within
x^2 + y^2 > 1 and those points within
x^2 + y^2 <= 1.  The ratio of these counts should
be ~PI.


Created on May 14, 2019

@author: kyle
'''
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Enough digits to compare our estimate to
PI = 3.14159265359
OUTCOMES = ['inside', 'outside']

def sample():
    x = random.random() # returns [0,1)
    y = random.random() # returns [0,1)
    value = pow(x,2) + pow(y, 2)
    
#     print(f'x={x}, y={y}, value={value}')
    category = 'none'
    
    if value <= 1:
        category = 'inside'
    else:
        category = 'outside'    
    return x, y, category

def run():
    print('Inside findpi.run()')
    # First we initialize our state variables
    count = 0;
    n_inside = 0
    x_list = []
    y_list = []
    cat_list = []
    
    end_count = 1000000
    
    while count < end_count:
        # make sure to increment
        count += 1
        
        #simulate
        (x, y, cat) = sample()
        
        # increment the correct counter
        if cat == 'inside':
            n_inside += 1
        
        # store values
        x_list.append(x)
        y_list.append(y)
        cat_list.append(cat)
        
    print(f'Number of values inside: {n_inside}')
    estimation_of_pi = n_inside/end_count * 4
    print(f'Estimation of PI: {estimation_of_pi}') # multiply by 4 b/c only sampling 1/4 of unit cirle
    print(f'Percent error: {abs((estimation_of_pi/PI) - 1) * 100}%')
    