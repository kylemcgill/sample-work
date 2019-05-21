'''
This module is to simulate stock price paths.
Maybe have the option to have a markov chain 
included? idk...

Created on May 14, 2019

@author: kyle
'''

import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MIN_PRICE = 0.0

def plot_price_paths(price_paths):
    # Convert the data gathered into DataFrame
    df = pd.DataFrame

# First we need to have a generator for the random walk
def generate_return(price, mu, sigma):
    '''
    This method accepts the price at time=t
    and returns a new price at time=t+1.
    generate_return samples a gaussian  distribution 
    for the return between time t and t+1.
    
    price: The price at time=t
    mu: The average return for a timestep. Could be considered the drift, usually 0.
    sigma: The standard deviation of the return for a timestep
    
    RETURN: Price at time=t+1
    '''
    
    new_price = price + price * random.gauss(mu, sigma)
    
    return new_price

def run(n_timesteps, n_paths):
    '''
    Creates random paths to simulate a security
    
    n_timesteps: Number of timesteps to simulate until we stop
    n_paths: The number of price paths we would like to generate
    '''
    
    opening_price = 100.0
    sd = 0.01
    
    price_path = [[] for i in range(0, n_paths)]
    
    
    for path in range(0, n_paths):
        # Reinitialize the path variables
        current_price = opening_price

        for timestep in range(0, n_timesteps):
            
            current_price = generate_return(current_price, 0.0, sd)
            
            # Can't let the price go below 0.0
            price_path[path].append(max(current_price, 0.0)) # append the current_price to our current path
            
    
    print(f'finished generating price paths.')
    for path in price_path:
        print(f'Path: {path}')
    
    plot_price_paths(price_path)
    
            
            
            