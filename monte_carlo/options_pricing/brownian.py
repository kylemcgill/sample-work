'''
This module creates price paths using brownian 
motion given a mean and a standard deviation.

Makes use of pandas DataFrame to store the lists
generated.

Created on May 16, 2019

@author: kyle
'''
import random
import pandas as pd
import numpy as np


def sample(current_price, mu, sd):
    return current_price + current_price * random.normalvariate(mu, sd)

def generate_price_paths(starting_price, mu, sd, n_timesteps, n_paths):
    # Check to make sure we put in good values
    assert starting_price > 0, f'Starting price is less than or equal to 0. starting_price: {starting_price}'
    assert starting_price != None, 'starting_price = None'
    assert mu != None, 'mu = None'
    assert sd != None, 'sd = None'
    assert n_timesteps != None, 'n_timesteps = None'
    assert n_timesteps > 0, f'n_timesteps <= 0. n_timesteps={n_timesteps}'
    assert n_paths > 0, f'n_paths <= 0. n_paths = {n_paths}'
    
    current_timestep = 0
    current_price = starting_price
    
    current_path = 0;
    
    
    column_names = [f'path_{i}' for i in range(0,n_paths)]
    df = pd.DataFrame(columns=column_names) 

    while current_path < n_paths:
            
        
        # Reinit all the variables
        prices = [0 for i in range(0,n_timesteps)]
#         print(prices)
        
        current_timestep = 0
        current_price = starting_price
        prices[current_timestep] = current_price
        
        while current_timestep < n_timesteps-1: #-1 for the starting price
            current_timestep += 1 # Since we already set the starting price
            
            new_price = sample(current_price, mu, sd);
            
#             print(f'Current timestep: {current_timestep}')
            prices[current_timestep] = new_price
            
            current_price = new_price
        
#         print(prices)    
        df[f'path_{current_path}'] = prices
#         print('-'*40, '\n', df)

        current_path += 1
    
    # Print out the df just to be sure
    df.head()
    
    return df