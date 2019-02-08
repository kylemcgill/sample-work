# sample-work

The purpose of this repository is to showcase my development skills in python. All work shown here is my own with only common libraries (scikit-learn, pandas, numpy, scipy, etc.) used.  If you are a lurker, feel free to poke around; although, I'm sure nothing done here is ground breaking.

*Microsoft Malware Detection* 

This section uses the Kaggle: Microsoft Malware Prediction competition dataset which can be found here <https://www.kaggle.com/c/microsoft-malware-prediction/data>. This dataset has a non-trivial number of NaN values which force the Data Scientist to account for these in some way.  My intuition for developing this model is to think about how malicious hackers would attack a pool of machines. They would most likely: (1) Exploit a software security exploit or (2) Execute a phishing attack aimed at those machines with a high probability of success.  This means we want to try and find groups of machines which are most likely to be attacked.  The first point should be easy enough to find, for some version of software there will be a high occurance of 'HasDetections.' The second is a little nuansed, what type of person is likely to succumb to a phishing attack, and can the specs of the machine give us insight into this person? 

Instuctions on running the micrsoft_malware_detection package can be found in ./kaggle/microsoft_malware_detection/README

**Future Work**

This repository is expected to grow over time as I attempt new public challenges. The hope is to attempt a variety of different datasets to develop my skills.