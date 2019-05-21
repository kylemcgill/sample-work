'''
This module creates price paths given a text file 
of some time series of prices. It generates a list of 
returns for each of the time steps and then randomly 
samples without replacement. The idea here being we 
assume these real returns could happen, just in a 
different order.


Created on May 16, 2019

@author: kyle
'''

import random
import pandas as pd
import numpy as np

def load_file(filename):
    '''
    Reads in the file and creates an array of prices out of it
    in ascending date order.
    '''
    
    # I know that my file has prices listed as [most_recent, least_recent]
    # Need to flip the elements
    historical_path = []
    with open(filename, 'r') as fin:
        for line in fin:            
            to_float = float(line)
#             print(f'As Float: {to_float}')
            historical_path.append(to_float)
#             print(historical_path)
#             print('\n')
    historical_path.reverse()
    return historical_path

def generate_stepped_returns(historical_path):
    return [historical_path[i+1]/historical_path[i] for i in range(0, len(historical_path)-1)]

def sample_with_replacement(stepped_returns):
    pass

def sample_without_replacement(stepped_returns):
    random.shuffle(stepped_returns)
#     print('stepped_returns: ', stepped_returns)
#     print('new_returns: ', new_returns)
    return stepped_returns

def get_new_price(current_price, percent_return):
    return current_price * percent_return

def generate_price_paths(filename, n_timesteps, n_paths, with_replacement=False):
    historical_path = load_file(filename) # raw prices
    print(historical_path)
    historical_stepped_returns = generate_stepped_returns(historical_path) # %returns in the historical time series; size=len(historical_prices)-1
    print(historical_stepped_returns)
    
    # Check some boundry conditions
    assert historical_stepped_returns != None, 'historical_stepped_returns = None!'
    assert len(historical_path) == n_timesteps, f'Number of historical prices is not equal to number of timesteps. len(history)={len(historical_path)}, n_timesteps={n_timesteps}'
    
    column_names = [f'path_{i}' for i in range(0,n_paths)]
    df = pd.DataFrame(columns=column_names)
    
    current_timestep = 0
    current_path = 0
    starting_price = historical_path[0]
    print(f'Starting price: {starting_price}, type={type(starting_price)}')
    
    while current_path < n_paths:
        current_timestep = 0
        prices = [0 for i in range(0,n_timesteps)]
        current_price = starting_price
        prices[current_timestep] = current_price
        
        # precompute the samples now b/c easier to read and faster
        sampled_returns = []
        if with_replacement:
            sampled_returns = sample_with_replacement(historical_stepped_returns)
        else:
            sampled_returns = sample_without_replacement(historical_stepped_returns)
#             print('sampled_returns:', sampled_returns)
#         print(f'with_replacement: {with_replacement}')
#         print('sampled_returns:', sampled_returns)
        # sampling loop
        while current_timestep < n_timesteps-1: # -1 for the starting price already 
            new_price = get_new_price(current_price, sampled_returns[current_timestep])
            
            current_timestep += 1 # increment here because sampled_returns needs to know the current_timestep, not the next timestep
            prices[current_timestep] = new_price
            
            current_price = new_price
        
        print(f'length of prices: {len(prices)}')
        df[f'path_{current_path}'] = prices
        
        current_path += 1
    
    return df
    