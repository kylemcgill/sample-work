# sample-work

The purpose of this repository is to showcase my development skills in python. All work shown here is my own with only common libraries (scikit-learn, pandas, numpy, scipy, etc.) used.  If you are a lurker, feel free to poke around; although, I'm sure nothing done here is ground breaking.

**Housing Price Prediction**

As of right now, this sample project scores in the Top 51% of the [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) without doing much pre-processing. Since the housing prices are inherintly skewed due to the fact they can't be sold below 0 and in theory they could be sold for an infinite sum, the model predicts the log(sale_price). Future work for pre-processing includes looking for outliers in the data set as well as handling `NaN` values better. Currently, `NaN` values are replaced with the mode of a categorical column, or the mean for numerical columns.  I also have not corrected for skew for any of the numerical columns which might improve prediction.

 As of the time of this writing, I use a Kernal Ridge Regression simply because the cross validation look best for this model. I have not played around with different `alpha` with this model, but it is on the todo list.

**Monte Carlo**

A project centered around learning the monte carlo method to approximate integrals as well as learning how to use the method to simulate options pricing using simple models.  

*coinflip*: The most simple monte carlo method for simulating a coin flip `N` times.
*findpi*: Using the monte carlo method to estimate `PI` by randomly sampling the first quadrant of an inscribed unit circle.
*stockprice*: Uses Brownian Motion to generate random price walks. In this example, the starting price and subsequent drft and variance were chosen at random.

**opencv Tutorials**

This repository is for the C++ implementation of the opencv tutorials to learn more about computer vision. This project is last on the priority list as it aims as a sandbox for practicing with a new tool.


**Future Work**

Due to the terms and conditions for most of the Kaggle competitions being restrictive with the data, I am unlikely to continue posting that work here.  I will, however, try and expand my breadth of knowledge with toy problems in order to keep my skills fresh.  
