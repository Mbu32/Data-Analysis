import math
import os
import random
from pathlib import Path
from collections import defaultdict
from itertools import product
import pickle

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
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
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from pygam import LogisticGAM, s, f, l

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from dmba import plotDecisionTree, textDecisionTree, classificationSummary

from pandas import DataFrame
def rec_prec_f1(TN,FP,FN,TP):
    precision = TP/ (TP + FP)
    recall = TP/(TP+FN)
    specificity = TN / (TN + FP)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    print(f'Precision: {precision} and recall:{recall}, specificity:{specificity} and lastly f1: {f1_scores}')
    return()

data = pd.read_excel('Data/default_crdtcard.xls',skiprows=1)


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

#extra inputs for marriage and education. Will fix to include extras into the other category already provided. Also turn into categorical data
data['MARRIAGE'] = data['MARRIAGE'].replace({0:3}).astype('category')
data['EDUCATION'] = data['EDUCATION'].replace({0:4, 5:4, 6:4}).astype('category')



#Lets group the ages. more people in the younger section and thus made the groups smaller young and wider for older. 
#[(21, 23] < (23, 26] < (26, 29] < (29, 32] ... (40, 45] <(45, 55] < (55, 65] < (65, 80]]

bins = [21,23,26,29,32,35,40,45,55,65,80]
cdata=data.copy()
age_categories = pd.cut(cdata.AGE,bins)
cdata.AGE = pd.cut(cdata.AGE,bins)

dummies = ['AGE','SEX','EDUCATION','MARRIAGE']
x= pd.get_dummies(cdata[dummies], drop_first=True,dtype=int)

d_copy = cdata.drop(columns=dummies)
data_processed = pd.concat([d_copy,x],axis=1)

features = ['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
       'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
       'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5',
       'PAY_AMT6', 'SEX', 'AGE_(23, 26]',
       'AGE_(26, 29]', 'AGE_(29, 32]', 'AGE_(32, 35]', 'AGE_(35, 40]',
       'AGE_(40, 45]', 'AGE_(45, 55]', 'AGE_(55, 65]', 'AGE_(65, 80]',
       'EDUCATION_2', 'EDUCATION_3', 'EDUCATION_4', 'MARRIAGE_2',
       'MARRIAGE_3']

features_updated = ['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
       'BILL_AMT1', 'BILL_AMT6', 
       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5',
       'PAY_AMT6', 'SEX', 'AGE_(23, 26]',
       'AGE_(26, 29]', 'AGE_(29, 32]', 'AGE_(32, 35]', 'AGE_(35, 40]',
       'AGE_(40, 45]', 'AGE_(45, 55]', 'AGE_(55, 65]', 'AGE_(65, 80]',
       'EDUCATION_2', 'EDUCATION_3', 'EDUCATION_4', 'MARRIAGE_2',
       'MARRIAGE_3']


outcome = 'default payment next month'
X=data_processed[features_updated]
y=data_processed[outcome]

vif = DataFrame()
vif['feature']=X.columns
vif['VIF']=[variance_inflation_factor(X.values,i)
            for i in range(X.shape[1])]


#print(vif)
'''
 feature       VIF
0      LIMIT_BAL  4.317632
1          PAY_0  1.916777
2          PAY_2  3.209287
3          PAY_3  3.705843
4          PAY_4  4.428398
5          PAY_5  4.965625
6          PAY_6  3.455632
7      BILL_AMT1  4.651830
8      BILL_AMT6  5.134526
9       PAY_AMT1  1.347737
10      PAY_AMT2  1.262896
11      PAY_AMT3  1.310507
12      PAY_AMT4  1.283831
13      PAY_AMT5  1.321244
14      PAY_AMT6  1.236849
15           SEX  8.140594
16  AGE_(23, 26]  2.178476
17  AGE_(26, 29]  2.493831
18  AGE_(29, 32]  2.240301
19  AGE_(32, 35]  2.106639
20  AGE_(35, 40]  2.565165
21  AGE_(40, 45]  2.154725
22  AGE_(45, 55]  2.221147
23  AGE_(55, 65]  1.235402
24  AGE_(65, 80]  1.032575
25   EDUCATION_2  2.371086
26   EDUCATION_3  1.628008
27   EDUCATION_4  1.049605
28    MARRIAGE_2  2.466051
29    MARRIAGE_3  1.035899
'''
#Check linearity
bt_shift = [
    'BILL_AMT1', 'BILL_AMT6'
]
X_bt = X.copy()
bt_safe = [
    'LIMIT_BAL',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
    'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]
for col in bt_safe:
    X_bt[col + '_log'] = X[col] * np.log(X[col] + 1)

for col in bt_shift:
    shift = abs(X[col].min()) + 1
    X_bt[col + '_log'] = X[col] * np.log(X[col] + shift)
    
X_bt_sm = sm.add_constant(X_bt)

bt_model = sm.Logit(y, X_bt_sm)
bt_result = bt_model.fit(disp=False)

#print(bt_result.summary())

'''
LIMIT_BAL_log 0.077
PAY_AMT1_log  0.000
PAY_AMT2_log  0.000
PAY_AMT3_log  0.015
PAY_AMT4_log  0.008
PAY_AMT5_log  0.000
PAY_AMT6_log  0.000
BILL_AMT1_log 0.000
BILL_AMT6_log 0.183


pvalue<.05  the linearity is violated
pvalue>= .05   its fine

only two are not significant: 
limit_bal
Bill_amt6
'''

X_train,X_test, y_train,y_test = train_test_split(
    X,y,test_size=.5,random_state=42,stratify=y
)

#need to separate out categories that can be scaled/cant be scaled
scale_features = ['LIMIT_BAL',
       'BILL_AMT1', 'BILL_AMT6', 
       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5',
       'PAY_AMT6']
non_scale_features = ['SEX', 'AGE_(23, 26]',
       'AGE_(26, 29]', 'AGE_(29, 32]', 'AGE_(32, 35]', 'AGE_(35, 40]',
       'AGE_(40, 45]', 'AGE_(45, 55]', 'AGE_(55, 65]', 'AGE_(65, 80]',
       'EDUCATION_2', 'EDUCATION_3', 'EDUCATION_4', 'MARRIAGE_2',
       'MARRIAGE_3', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',]


preprocessor = ColumnTransformer([
    ('scaler', StandardScaler(), scale_features),      
    ('passthrough', 'passthrough', non_scale_features)  
])


X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Convert back to DataFrames with column names
feature_names = scale_features + non_scale_features
X_train_f = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
X_test_f = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)

'''
Attempted KNN,
after CV, with optimal k of 3 and a resulting f1 of .391
optimal threshold of .2 with an f1 of .42
decided to leave out KNN, and not include even as a feature. Simply adding noise.

Also used SMOTE(), was simply helping overfit training data F1 score on train data high, 
test data extremely low
F1 score on test data increased dramatically after removing.

Will begin with LogisticRegression for reference then GAM

and then continue with possibly randomforrest or XGboost
'''

LR= LogisticRegression(max_iter=1000,
                       C=1,
                       class_weight='balanced'
                       ) 

LR.fit(X_train_f,y_train)
y_prob=LR.predict_proba(X_test_f)[:,1]
y_class=LR.predict(X_test_f)

#Finding optimal threshold
best_f1=0
best_thresh=.5

y_prob1 = cross_val_predict(LR,X_train_f,y_train,cv=5,method='predict_proba')[:,1]

for thresh in np.arange(.1,.6,.05):
    y_pred=(y_prob1>=thresh).astype(int)
    f1=f1_score(y_train,y_pred)
    if f1 > best_f1:
        best_f1=f1
        best_thresh=thresh
#print(f'Optimal threshold: {best_thresh}')

y_prob_optimal = LR.predict_proba(X_test_f)[:,1]
y_test_optimal = (y_prob_optimal>= best_thresh).astype(int)

#confusionMatrix

cm=confusion_matrix(y_test,y_test_optimal)
TN,FP,FN,TP=  cm.ravel()

#rec_prec_f1(TN,FP,FN,TP)
#auc_test= roc_auc_score(y_test,y_test_optimal)
#print(f'for regression AUC: {auc_test}')
'''
before optimal Threshold:
Precision: 0.3770610617865555 and recall:0.6271850512356841, 
specificity:0.7057010785824345 and lastly f1: 0.47097431254950767
AUC = 0.6664430649090592


After Optimal Threshold of: 0.5500000000000002
Precision: 0.4833099579242637 and recall:0.5192887281494877, 
specificity:0.842321520287622 and lastly f1: 0.5006537846869098
for regression AUC: 0.6808051242185548
Lets try GAM & then compare.
'''

#original features for GAM, looking for predictability, will be keeping correlated features, 
# Laptop couldnt handle GAM calculation... Will rerun when possible.


'''
terms = (
    s(0,n_splines=10) +           # LIMIT_BAL
    f(1) +           # SEX
    f(2) +           # EDUCATION
    f(3) +           # MARRIAGE
    s(4,n_splines=10) +           # AGE
    l(5) + l(6) + l(7) + l(8) + l(9) + l(10) +  # PAY_0 to PAY_6
    l(11) + l(12) + l(13) + l(14) + l(15) + l(16) +  # BILL_AMT1-6
    l(17) + l(18) + l(19) + l(20) + l(21) + l(22)     # PAY_AMT1-6

    decreasing to ease up on memory...
)


gam = LogisticGAM(terms)

gam.gridsearch(X_train_f.values,y_train.values)   Couldn't compute on my laptop. Cant tune lambda. will use default

print(gam.summary())

#y_predict=gam.predict(X_test_f.values)

gam.fit(X_train_f.values, y_train.values)

y_prob = gam.predict_proba(X_test_f.values)
best_f1 = 0
best_thresh = 0.5
for t in np.arange(0.1, 0.9, 0.05):
    y_pred = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t
print(f"Optimal threshold: {best_thresh}")
y_predict=(y_prob>= best_thresh).astype(int)

#confusion Matrix + AUC
cm=confusion_matrix(y_test,y_predict)
TN,FP,FN,TP=  cm.ravel()
rec_prec_f1(TN,FP,FN,TP)

auc_test= roc_auc_score(y_test,y_prob)
print(f'for GAM AUC: {auc_test:.3f}')
'''





#Random Forest Implementation
features_updated = ['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
       'BILL_AMT1', 'BILL_AMT6', 
       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5',
       'PAY_AMT6', 'SEX', 'AGE_(23, 26]',
       'AGE_(26, 29]', 'AGE_(29, 32]', 'AGE_(32, 35]', 'AGE_(35, 40]',
       'AGE_(40, 45]', 'AGE_(45, 55]', 'AGE_(55, 65]', 'AGE_(65, 80]',
       'EDUCATION_2', 'EDUCATION_3', 'EDUCATION_4', 'MARRIAGE_2',
       'MARRIAGE_3']

outcome = 'default payment next month'


rf=RandomForestClassifier(n_estimators=500,
                          random_state=42,
                          oob_score=True)

X_train2 = X_train[['PAY_0','AGE_(55, 65]','EDUCATION_4','AGE_(65, 80]']]
X_test2=X_test[['PAY_0','AGE_(55, 65]','EDUCATION_4','AGE_(65, 80]']]

rf.fit(X_train,y_train)


y_prediction_rf = rf.predict(X_test)
test_accuracy = accuracy_score(y_test,y_prediction_rf)
#print(test_accuracy)
#0.8136


t_oob = rf.oob_decision_function_
oob_pred = t_oob.argmax(axis=1)
oob_accuracy = (oob_pred ==y_train).mean()

#print(oob_accuracy) 
#0.8188666666666



#Feature importance
'''
result = permutation_importance(rf,X_test,y_test,n_repeats=10,random_state=42)
perm_importance = DataFrame({
    'feature':features_updated,
    'Permutation Importance':result.importances_mean,
    'Std':result.importances_std
}).sort_values('Permutation Importance',ascending=True)

with open('Data/rf_importance.pkl','wb') as f:
    pickle.dump(perm_importance,f)
'''

with open("Data/rf_importance.pkl", "rb") as f:  
    loaded_importance = pickle.load(f)

#print('\n Permutation importance:')
#print(loaded_importance.sort_values('Permutation Importance', ascending=False))


'''
 feature  Permutation Importance       Std
1          PAY_0                0.056580  0.001447
23  AGE_(55, 65]                0.000287  0.000191
27   EDUCATION_4                0.000113  0.000090
24  AGE_(65, 80]                0.000047  0.000043
29    MARRIAGE_3               -0.000080  0.000098
26   EDUCATION_3               -0.000167  0.000269
19  AGE_(32, 35]               -0.000327  0.000162
25   EDUCATION_2               -0.000327  0.000561
16  AGE_(23, 26]               -0.000347  0.000240
17  AGE_(26, 29]               -0.000453  0.000350
18  AGE_(29, 32]               -0.000473  0.000313
2          PAY_2               -0.000667  0.000533
21  AGE_(40, 45]               -0.000673  0.000312
22  AGE_(45, 55]               -0.000713  0.000289
4          PAY_4               -0.000887  0.000403
20  AGE_(35, 40]               -0.001020  0.000417
28    MARRIAGE_2               -0.001053  0.000610
15           SEX               -0.001280  0.000292
3          PAY_3               -0.001380  0.000331
6          PAY_6               -0.001473  0.000224
8      BILL_AMT6               -0.001500  0.000781
0      LIMIT_BAL               -0.001713  0.000693
13      PAY_AMT5               -0.001773  0.000750
7      BILL_AMT1               -0.001827  0.000884
11      PAY_AMT3               -0.001867  0.000399
5          PAY_5               -0.002053  0.000511
9       PAY_AMT1               -0.002513  0.000712
10      PAY_AMT2               -0.002587  0.000398
12      PAY_AMT4               -0.002713  0.000572
14      PAY_AMT6               -0.002907  0.000477

We can see here that PAY_0 plays a huge rule in predicting, but some of the others are simply adding noise. Going back and adjusting features
'''


#trying with varying features.
'''
rf1=RandomForestClassifier(n_estimators=500,
                          random_state=42,
                          oob_score=True)
X_train2 = X_train[['PAY_0','AGE_(55, 65]','EDUCATION_4','AGE_(65, 80]']]
X_test2=X_test[['PAY_0','AGE_(55, 65]','EDUCATION_4','AGE_(65, 80]']]
rf1.fit(X_train2,y_train)
y_prediction_rf = rf1.predict(X_test2)
test_accuracy_2 = accuracy_score(y_test,y_prediction_rf)
print(f'Test accuracy with only positively contributing features {test_accuracy_2}')
'''
#Test accuracy with only positively contributing features 0.8184666666666667
'''
#top 2 features
rf2=RandomForestClassifier(n_estimators=500,
                          random_state=42,
                          oob_score=True)
X_train3 = X_train[['PAY_0','AGE_(55, 65]']]
X_test3=X_test[['PAY_0','AGE_(55, 65]']]
rf2.fit(X_train3,y_train)
y_prediction_rf = rf2.predict(X_test3)
test_accuracy_3 = accuracy_score(y_test,y_prediction_rf)
print(f'Test accuracy with only 2 contributing features {test_accuracy_3}')
'''
#Test accuracy with only 2 contributing features 0.8178
'''
rf3=RandomForestClassifier(n_estimators=500,
                          random_state=42,
                          oob_score=True)
X_train3 = X_train[['PAY_0']]
X_test3=X_test[['PAY_0']]
rf3.fit(X_train3,y_train)
y_prediction_rf = rf3.predict(X_test3)
test_accuracy_4 = accuracy_score(y_test,y_prediction_rf)
print(f'Test accuracy with only 1 contributing features {test_accuracy_4}')
'''
#Test accuracy with only Pay_0: 0.8178666666666666
'''
Comparing all three
1) with all we had accuracy of 0.8136

2) with only positive: 0.8184666666666667

3)2 contributing features: 0.8178

4)Test accuracy with only 1 contributing features 0.8178666666666666

Option 2 being the highest tells us: from option 2 to 3 we dropped about .0006. meaning thats how much those 2 extra features were contributing.

Pay_0 contributing: 0.8178666666666666 of the accuracy to the model! 

'''



'''
we could do a GridsearchCV to tune some parameters but my computer wont handle all that but somethings we could try:


param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': [None, 'balanced', {0:1, 1:2}, {0:1, 1:3}]
}

rf_base = RandomForestClassifier(oob_score=True, random_state=42)
grid_search = GridSearchCV(rf_base, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train2, y_train)
'''








#lastly, we have XGBoost
X_train_xgb = X_train[['PAY_0', 'AGE_(55, 65]', 'EDUCATION_4', 'AGE_(65, 80]']].values
X_test_xgb = X_test[['PAY_0', 'AGE_(55, 65]', 'EDUCATION_4', 'AGE_(65, 80]']].values

xgb_default = XGBClassifier(
    objective='binary:logistic',
    n_estimators=200,
    max_depth=3,
    reg_lambda=1,
    learning_rate=0.1,
    subsample=.8,
    eval_metric='error',
    random_state=42,
    early_stopping_rounds=20  
)


xgb_default.fit(
    X_train_xgb, y_train,
    eval_set=[(X_test_xgb, y_test)],
    verbose=True
)
# Predictions
y_pred = xgb_default.predict(X_test_xgb)
y_pred_proba = xgb_default.predict_proba(X_test_xgb)[:, 1]

# Evaluation
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"\nBest iteration: {xgb_default.best_iteration}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

