# sample-work

The purpose of this repository is to showcase my development skills in python. All work shown here is my own with only common libraries (scikit-learn, pandas, numpy, scipy, etc.) used.  If you are a lurker, feel free to poke around; although, I'm sure nothing done here is ground breaking.

**Housing Price Prediction**

As of right now, this sample project scores in the Top 15% of the [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). Since the housing prices are inherintly skewed due to the fact they can't be sold below 0 and in theory they could be sold for an infinite sum, the model predicts the log(sale_price).

 As of the time of this writing, I use a Kernal Ridge Regression simply because the cross validation look best for this model. You can see I search for a better alpha, a better algorithm would be to take the best cross validation scores and search closer around those `alpha`s which look promising. A jump was made from Top 42% to Top 15% in one day by utlilizing feature creation and outlier removal. 
 
 Future work would be to weight the neighborhoods differently than just as a category since there is correlation between the distributions of each neighborhood.  An example would be assigning a numerical values in order of the relative median/mean of each neighborhood, or even better assigning the weight of each neighborhood as it relates to the increase of sale price.  An example would be: If neighborhood A had a median that was 2 times as expensive as neighborhood B, then then the ratio of weights (A/B) should equal 2. We could then choose {A=2, B=1} or {A=4, B=2} etc.

**Monte Carlo**

A project centered around learning the monte carlo method to approximate integrals as well as learning how to use the method to simulate options pricing using simple models.  

*coinflip*: The most simple monte carlo method for simulating a coin flip `N` times.

*findpi*: Using the monte carlo method to estimate `PI` by randomly sampling the first quadrant of an inscribed unit circle.

*stockprice*: Uses Brownian Motion to generate random price walks. In this example, the starting price and subsequent drft and variance were chosen at random.

**opencv Tutorials**

This repository is for the C++ implementation of the opencv tutorials to learn more about computer vision. This project is last on the priority list as it aims as a sandbox for practicing with a new tool.


**Future Work**

Due to the terms and conditions for most of the Kaggle competitions being restrictive with the data, I am unlikely to continue posting that work here.  I will, however, try and expand my breadth of knowledge with toy problems in order to keep my skills fresh.  
