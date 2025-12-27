from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression , LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score

import statsmodels.api as sm

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from pygam import LinearGAM, s, f, l


from dmba import classificationSummary

import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_excel('Data/default_crdtcard.xls',skiprows=1)
#print(data.info())
#print(data.head(n=2))
#print(data.columns)

'''
Columns/Features 23 in total.
'ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
'default payment next month'
       
This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
X2: Gender (1 = male; 2 = female).
X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
X4: Marital status (1 = married; 2 = single; 3 = others).
X5: Age (year).
X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005. 
X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.


to turn into dummy variables:
Age
Gender - to keep it consistent
Education
Marital status

'''
#print(data['MARRIAGE'].value_counts())
#print(data['EDUCATION'].value_counts())

#extra inputs for marriage and education. Will fix to include extras into the other category already provided. Also turn into categorical data

data['MARRIAGE'] = data['MARRIAGE'].replace({0:3}).astype('category')
data['EDUCATION'] = data['EDUCATION'].replace({0:4, 5:4, 6:4}).astype('category')




#print(data['AGE'].min())
#print(data['AGE'].max())
#print(data['AGE'].median())

#Lets group the ages. more people in the younger section and thus made the groups smaller young and wider for older. 
#[(21, 23] < (23, 26] < (26, 29] < (29, 32] ... (40, 45] <(45, 55] < (55, 65] < (65, 80]]

bins = [21,23,26,29,32,35,40,45,55,65,80]

age_categories = pd.cut(data.AGE,bins)
data.AGE = pd.cut(data.AGE,bins)
#print(data.dtypes)

dummies = ['AGE','SEX','EDUCATION','MARRIAGE']



#lets turn into dummies

x= pd.get_dummies(data[dummies], drop_first=True,dtype=int)

d_copy = data.drop(columns=dummies)
data_processed = pd.concat([d_copy,x],axis=1)

print(data_processed.columns)

print(data['MARRIAGE'].value_counts())
print(data['EDUCATION'].value_counts())