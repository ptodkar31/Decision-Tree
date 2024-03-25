# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 23:20:39 2024

@author: Priyanka
"""

"""
In the recruitment domain, HR faces the challenge
of predicting if the candidate is faking their salary 
or not. For example, a candidate claims to have 5 years
of experience and earns 70,000 per month working as
a regional manager. The candidate expects more money
than his previous CTC. We need a way to verify their 
claims (is 70,000 a month working as a regional manager with an
experience of 5 years a genuine claim or does he/she make less than that?)
Build a Decision Tree and Random Forest model with monthly income as the target variable

Business Problem
Q.What is the business objective?
The Need to Make Quick Hire One of the major struggles that the recruiters 
these days encounter is the requirement of resources to be hired in a very 
short span of time.With very little time candidates are made to go 
through multiple rounds screening and interviews, but the need of good 
calibre and talent is very important at the same time. 

Importance of Validating/ verifying -
Credentials One of the most common issues we see in the recruitment process
is that the candidates not being able to submit the credentials or academic 
validations in relation to the roles they became eligible to apply for the 
jobs at hand.It takes a lot on the part of the recruitment team to actually 
put across clearly the need for the validations to ensure the authenticity 
of the candidature.
 
Benchmarking Salaries / Remuneration-
It is usually a tricky situation as to whether a benchmarking of the present 
compensation of the candidate or irrespective of the past remuneration,
an entirely new talent led offer is rolled out.Also this may differ from 
candidate to candidate -Fresher or one with past experience in the same field.
The negotiation process that goes on is vital as past that, the offer is rolled out. 
The acceptance of that offer by the candidate is a persuasion that HR
 
Q.Are there any constraints?
Staffing in today’s day and age is a unique task and very few companies get it right. 
Ours is a large economy and there is currently a demand and supply disequilibrium 
in most domains- be it the IT industry, Engineering /Manufacturing,
Research, Media and Entertainment, Healthcare etc.

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
HR=pd.read_csv("C:\Data Set\HR_DT.csv")

#exploratory data analysis
HR.dtypes
HR.describe()
#Average  
#no of Years of Experience of employee   74194.92 
#minimum is 37731 and max is 122391

#Let us rename the columns
new_names = ['Position','No_Exp','Month_income']
df = pd.read_csv(
    "C:\Data Set\HR_DT.csv", 
    names=new_names,           # Rename columns
    header=0,                  # Drop the existing header row
    usecols=[0,1,2],       # Read the first 3 columns
    )
df.dtypes
plt.hist(df.No_Exp)
#Data is almost normally distributed
plt.hist(df.Month_income)
#Data is apperently normally distributed

#let us check outliers
plt.boxplot(df.No_Exp)
#There are  no outliers 
plt.boxplot(df.Month_income)
#There are no outliers

df.isnull().sum()
#There are no null values

#Data preprocessing

bins = [35000,60000,90000,110000,122391]
group_name=["first_range", "second_range","third_range","fourth_range"]
df['Month_income']=pd.cut(df['Month_income'],bins,labels=group_name)
#let us first check the null values after conversion
df.isnull().sum()

from sklearn.preprocessing import LabelEncoder

lb=LabelEncoder()
df["Position"]=lb.fit_transform(df["Position"])
##There are several columns having different scale
#Normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_func(df.iloc[:,0:2])

#Let us check that how many unique values are the in the output column
df["Month_income"].unique()
df["Month_income"].value_counts()

#let us assign input features as predictors and output as target
colnames=list(df.columns)
predictors=colnames[0:2]
target=colnames[2]

#Splitting data into train and Test
from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.2)

#model bulding
from sklearn.tree import DecisionTreeClassifier as DT
model=DT(criterion='entropy')
model.fit(train[predictors],train[target])
preds=model.predict(test[predictors])
pd.crosstab(test[target],preds)
np.mean(preds==test[target])
#Let us check the accuracy on training dataset
preds=model.predict(train[predictors])
np.mean(preds==train[target])

# Now let us try for Random forest tree
from sklearn.ensemble import RandomForestClassifier
rand_for=RandomForestClassifier(n_estimators=500,n_jobs=1,random_state=42)
rand_for.fit(train[predictors],train[target])
from sklearn.metrics import accuracy_score,confusion_matrix
preds=model.predict(test[predictors])
pd.crosstab(test[target],preds)
np.mean(preds==test[target])

#Let us check the accuracy on training dataset
preds=model.predict(train[predictors])
np.mean(preds==train[target])
pd.crosstab(train[target],preds)
#this is again overfit model
rand_for=RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=20,criterion='gini')
rand_for.fit(train[predictors],train[target])
from sklearn.metrics import accuracy_score,confusion_matrix
preds=model.predict(test[predictors])
pd.crosstab(test[target],preds)
np.mean(preds==test[target])
