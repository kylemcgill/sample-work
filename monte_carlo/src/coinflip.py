'''
Created on May 14, 2019

@author: kyle
'''
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def flip():
    r = random.randint(1,100)
    # range() function is [lower, upper)
    if r in range(1,51):
        return 'heads'
    elif r in range(51,101):
        return 'tails'
    else:
        print(f'Random number out of range! number: {r}')
        exit()


def run(sims_to_run):
    # Initialize the start of the monte carlo
    count = 0
    n_heads = 0
    n_tails = 0
    end_count = sims_to_run
    
    while count < end_count:
        count += 1
        result = flip()
        if result == 'heads':
            n_heads += 1
        elif result == 'tails':
            n_tails += 1
    
    # At the end display the distribution
    print(f'Number of heads: {n_heads} -> {n_heads/end_count * 100}%')
    print(f'Number of tails: {n_tails} -> {n_tails/end_count * 100}%')
    
    # graph the distribution
    df = pd.DataFrame({
            'result' : ['heads', 'tails'],
            'counts' : [n_heads, n_tails]
        })
    sns.catplot(x='result', y='counts', kind='bar', data=df)
    plt.show()