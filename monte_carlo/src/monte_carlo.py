'''
The purpose of this file is to work on monte carlo simulations 
as they can be created in python

Created on May 14, 2019

@author: kyle
'''
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import coinflip
import findpi
import stockprice

BREAK_STR = '-'*40
print('starting monte carlo.')
# Start with flipping a coin

coinflip.run(100)
print(BREAK_STR)

findpi.run()
print(BREAK_STR)

stockprice.run(n_paths=1, n_timesteps=10)

print('done.')