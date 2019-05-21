import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(data, dtypes):
    sale_price = data['SalePrice']
    sale_price_str = 'SalePrice'
    columns = list(data)
    sns.set(style='darkgrid')
    
    for col in columns:
        print(f'Plot for: {col}')
        sns.relplot(x=col, y=sale_price_str, hue='Utilities', style='Utilities', data=data)
        plt.show() # code will block here until window has closed
