import math
import os
import random
from pathlib import Path
from collections import defaultdict
from itertools import product

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_predict
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, precision_recall_curve, make_scorer,f1_score,recall_score,precision_score
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from pygam import LinearGAM, s, f, l

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from dmba import plotDecisionTree, textDecisionTree, classificationSummary

from pandas import DataFrame

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
X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005.
The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
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


features = ['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
       'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
       'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5',
       'PAY_AMT6', 'SEX', 'AGE_(23, 26]',
       'AGE_(26, 29]', 'AGE_(29, 32]', 'AGE_(32, 35]', 'AGE_(35, 40]',
       'AGE_(40, 45]', 'AGE_(45, 55]', 'AGE_(55, 65]', 'AGE_(65, 80]',
       'EDUCATION_2', 'EDUCATION_3', 'EDUCATION_4', 'MARRIAGE_2',
       'MARRIAGE_3']

outcome = 'default payment next month'


#split data for test/train then ensure equal probabilities for default/no default to ensure less bias in model

X=data_processed[features]
y=data_processed[outcome]
# 1. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)



#lets scale our data.
scaler=StandardScaler()
X_trained_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


###### We're going to do KNN & CV, find best parameters
param_grid = {'n_neighbors':[3,5,7,9,11,13,15,17,19,21]}
knn_cv = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)
    
knn_cv.fit(X_trained_scaled,y_train)
best_k=knn_cv.best_params_['n_neighbors']
print(f'optimal k from cv:{best_k}, with an F1 of:{knn_cv.best_score_:.3f}')



knn = KNeighborsClassifier(n_neighbors=best_k) 
#to make sure we dont have any data leakage
knn_trainp = cross_val_predict(
    knn,
    X_trained_scaled,
    y_train,
    cv=5,
    method='predict_proba'
)[:, 1]


knn.fit(X_trained_scaled, y_train) 
knn_testp=knn.predict_proba(X_test_scaled)[:,1]



data_processed['borrower_score']=np.nan
data_processed['borrower_score'].loc[X_train.index,'borrower_score']=knn_trainp
data_processed['borrower_score'].loc[X_test.index,'borrower_score']=knn_testp


'''
#2 Apply SMOTE to the training set 
smote=SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

'''



















#logistic Regression
LR = LogisticRegression(max_iter=1000)
LR.fit(X_trained_scaled,y_train_res)

y_pred_prob = LR.predict_proba(X_test_scaled)[:,1]
y_pred_class = LR.predict(X_test_scaled)

#confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
TN, FP, FN, TP = cm.ravel()


def rec_prec_f1(TN,FP,FN,TP):
    precision = TP/ (TP + FP)
    recall = TP/(TP+FN)
    specificity = TN / (TN + FP)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    print(f'Precision: {precision} and recall:{recall}, specificity:{specificity} and lastly f1: {f1_scores}')
    return()


'''
precision :0.47885572139303484 ~ of all predicted defaults: 48% were actual defualts
recall:0.34810126582278483  ~of all actual defaulters, we caught 35%
specificity:0.8923985618900873 ~  all non defaulters we predicted 89% right woot wooot

f1 = 0.4031413612565445 ~ low because recall is low...


Threshold:
'''
y_pred_class = (y_pred_prob >= 0.3).astype(int)

cm = confusion_matrix(y_test, y_pred_class)
TN, FP, FN, TP = cm.ravel()


#rec_prec_f1(TN,FP,FN,TP)
'''
Precision: 0.3691210485736315 
and recall:0.5771549125979506,
specificity:0.7198253723677452 
and lastly f1: 0.4502703973665648
'''





vif = DataFrame()
vif["feature"] =  X.columns
vif['VIF'] =  [variance_inflation_factor(X.values,i)
               for i in range(X.shape[1])]
#print(vif)


'''
   feature        VIF
0      LIMIT_BAL   4.330758
1          PAY_0   1.917328
2          PAY_2   3.213951
3          PAY_3   3.728561
4          PAY_4   4.441534
5          PAY_5   4.987126
6          PAY_6   3.464758
7      BILL_AMT1  20.831599
8      BILL_AMT2  38.220042
9      BILL_AMT3  31.809041
10     BILL_AMT4  29.556822
11     BILL_AMT5  36.003837
12     BILL_AMT6  21.458256
13      PAY_AMT1   1.908595
14      PAY_AMT2   2.385728
15      PAY_AMT3   1.913540
16      PAY_AMT4   1.805446
17      PAY_AMT5   1.854983
18      PAY_AMT6   1.271150
19           SEX   8.143608
20  AGE_(23, 26]   2.178598
21  AGE_(26, 29]   2.493857
22  AGE_(29, 32]   2.240341
23  AGE_(32, 35]   2.106848
24  AGE_(35, 40]   2.565256
25  AGE_(40, 45]   2.154868
26  AGE_(45, 55]   2.221509
27  AGE_(55, 65]   1.235558
28  AGE_(65, 80]   1.032748
29   EDUCATION_2   2.372302
30   EDUCATION_3   1.628622
31   EDUCATION_4   1.050083
32    MARRIAGE_2   2.466663
33    MARRIAGE_3   1.036171
'''





#Next steps to improve : try a GLM model to interpret some coefficients:

X_train_scaled_df = pd.DataFrame(X_train, columns=X_train_res.columns)
X_train_scaled_df = sm.add_constant(X_train_scaled_df)

log_sm = sm.GLM(y_train,X_train_scaled_df.assign(const=1)
                ,family=sm.families.Binomial())
log_result = log_sm.fit()

#print(log_result.summary())


'''
GLM MODEL. alpha =0.05


                  coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
const           -0.6151      0.133     -4.620      0.000      -0.876      -0.354
LIMIT_BAL    -8.164e-07   2.29e-07     -3.567      0.000   -1.26e-06   -3.68e-07
PAY_0            0.6084      0.025     24.152      0.000       0.559       0.658
PAY_2            0.0849      0.028      2.978      0.003       0.029       0.141
PAY_3            0.0599      0.032      1.878      0.060      -0.003       0.122
PAY_4            0.0372      0.035      1.057      0.291      -0.032       0.106
PAY_5            0.0244      0.038      0.640      0.522      -0.050       0.099
PAY_6            0.0450      0.031      1.445      0.149      -0.016       0.106
BILL_AMT1    -4.198e-06   1.52e-06     -2.757      0.006   -7.18e-06   -1.21e-06
BILL_AMT2     9.416e-07   2.13e-06      0.443      0.658   -3.22e-06    5.11e-06
BILL_AMT3     3.591e-10   1.99e-06      0.000      1.000    -3.9e-06     3.9e-06
BILL_AMT4     1.091e-06    1.9e-06      0.574      0.566   -2.64e-06    4.82e-06
BILL_AMT5     1.637e-06      2e-06      0.818      0.413   -2.29e-06    5.56e-06
BILL_AMT6    -1.098e-06   1.54e-06     -0.712      0.477   -4.12e-06    1.93e-06
PAY_AMT1     -6.052e-06   2.67e-06     -2.266      0.023   -1.13e-05   -8.17e-07
PAY_AMT2     -1.094e-05   3.06e-06     -3.570      0.000   -1.69e-05   -4.93e-06
PAY_AMT3      -2.16e-06   2.28e-06     -0.946      0.344   -6.63e-06    2.31e-06
PAY_AMT4     -2.549e-06   2.35e-06     -1.082      0.279   -7.16e-06    2.07e-06
PAY_AMT5      5.261e-07   2.22e-06      0.237      0.813   -3.83e-06    4.88e-06
PAY_AMT6      -1.16e-06   1.71e-06     -0.680      0.497    -4.5e-06    2.18e-06
SEX             -0.0905      0.044     -2.066      0.039      -0.176      -0.005
AGE_(23, 26]    -0.1189      0.105     -1.138      0.255      -0.324       0.086
AGE_(26, 29]    -0.1472      0.104     -1.419      0.156      -0.350       0.056
AGE_(29, 32]    -0.1580      0.109     -1.451      0.147      -0.372       0.055
AGE_(32, 35]    -0.0376      0.111     -0.339      0.735      -0.255       0.180
AGE_(35, 40]    -0.0695      0.108     -0.645      0.519      -0.281       0.142
AGE_(40, 45]    -0.0895      0.113     -0.791      0.429      -0.311       0.132
AGE_(45, 55]     0.0532      0.111      0.478      0.632      -0.165       0.271
AGE_(55, 65]    -0.1986      0.167     -1.188      0.235      -0.526       0.129
AGE_(65, 80]     0.2040      0.394      0.518      0.605      -0.568       0.976
EDUCATION_2     -0.1003      0.051     -1.985      0.047      -0.199      -0.001
EDUCATION_3     -0.2252      0.069     -3.285      0.001      -0.360      -0.091
EDUCATION_4     -1.2699      0.285     -4.463      0.000      -1.828      -0.712
MARRIAGE_2      -0.2162      0.051     -4.237      0.000      -0.316      -0.116
MARRIAGE_3       0.0443      0.175      0.253      0.800      -0.299       0.388
================================================================================

have some features that arent contributing a lot: Since were doing a LogisticsR we're more so 
aiming for predictibabilty so not removing any features. but could in the future to see if it improves
predictibability. Some features could be helping a bit even if approaching insignificance
'''
