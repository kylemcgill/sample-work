'''
This is the main module for a simple options pricing monte 
carlo simulation. The question is really how do we want to 
display this process. Do we model brownian motion or do we 
do the random real return method? how about both and then 
show a comparison between the two!

1. For brownian motion we need to generate some paths using 
dS = (mu)*dt + (var)*dOmega

2. For historical data need to get the returns and then create 
a sampler to get each step and create the path.

3. Once we have paths for each method then we have some average 
return with some variance. Then we create an option price
based on different percentiles.

4. Create some visuals for the price paths and the resulting 
distributions.
'''

import brownian
import historical
import pandas as pd
import numpy as np
import seaborn as sns

BREAK_STR = '-'*40


print(f'Starting options pricing demo based on brownian motion and '
        'historical data of some selected securities...')

max_timesteps = 10
max_paths = 2

print(f'Generating brownian motion paths... ')
brownian_df = brownian.generate_price_paths(starting_price=100.0,
                                              mu=0.0,
                                              sd=0.005, 
                                              n_timesteps=max_timesteps,
                                              n_paths=max_paths)

print(brownian_df.head(3))
print(BREAK_STR)
print('Generating historical motion paths...')
historical_df = historical.generate_price_paths('/Users/kyle/Documents/workspace/monte_carlo/options_pricing/historical_data.txt', 
                                                n_timesteps=max_timesteps,
                                                n_paths=max_paths,
                                                with_replacement = False)

print(brownian_df.head(3))
print(BREAK_STR)
print('Creating distributions for both methods...')


print('Brownian Motion statistics:')
brownian_stats = brownian_df.describe()
print(brownian_df.describe())
print(BREAK_STR)
print('Historical Sampling statistics:')
historical_stats = historical_df.describe()
print(historical_df.describe())
print(BREAK_STR)

print('Displaying statistics from brownian and historical paths')
brownian_stats['mean'] = brownian_stats.mean(axis=1)
print(BREAK_STR)
print('Brownian stats with averages')
print(brownian_stats)
print(BREAK_STR)

print('Historical stats with averages')
historical_stats['mean'] = historical_stats.mean(axis=1)
print(historical_stats)

# draw_one_boxplot()


print(f'done.')


