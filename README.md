# sample-work

The purpose of this repository is to showcase my development skills in python. All work shown here is my own with only common libraries (scikit-learn, pandas, numpy, scipy, etc.) used.  If you are a lurker, feel free to poke around; although, I'm sure nothing done here is ground breaking.

**Housing Price Prediction**

As of right now, this sample project scores in the Top 51% of the [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) without doing much pre-processing. Since the housing prices are inherintly skewed due to the fact they can't be sold below 0 and in theory they could be sold for an infinite sum, the model predicts the log(sale_price). Future work for pre-processing includes looking for outliers in the data set as well as handling `NaN` values better. Currently, `NaN` values are replaced with the mode of a categorical column, or the mean for numerical columns.  I also have not corrected for skew for any of the numerical columns which might improve prediction.

 As of the time of this writing, I use a Kernal Ridge Regression simply because the cross validation look best for this model. I have not played around with different `alpha` with this model, but it is on the todo list.

**Future Work**

This repository is expected to grow over time as I attempt new public challenges. The hope is to attempt a variety of different datasets to develop my skills.