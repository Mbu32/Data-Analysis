#multiple linear regression
import numpy as np
import pandas as pd
import kaggle
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame,Series 
from scipy import stats
from sklearn.utils import resample
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression


import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence

from pygam import LinearGAM, s, l
from pygam.datasets import wage

from scipy.stats import bootstrap

from dmba import stepwise_selection
from dmba import AIC_score



from CleanCafe import cdata

'''
Aggregate transcation level data into daily revenue and then 
model it, enabling us to do some short horizon revenue forecasting

'''

ddata = cdata.copy()

#Turning data into daily data rather than per transaction ID
daily = (ddata.groupby(ddata['Transaction Date'].dt.date)
         .agg(
             DailyRevenue = ('Total Spent','sum'),
             TotalQuantity = ('Quantity','sum'),
             #NumTransactions=('Transaction ID','count') ,  collinearity 
         ).reset_index())


daily['Transaction Date'] = pd.to_datetime(daily["Transaction Date"])
daily['DayOfWeek']= daily['Transaction Date'].dt.dayofweek
daily['IsWeekend']= daily['DayOfWeek'].isin([5,6]).astype(int)

predictors = ['DayOfWeek','IsWeekend','TotalQuantity']

outcome = 'DailyRevenue'

cafe_lm = LinearRegression()
cafe_lm.fit(daily[predictors],daily[outcome])

print(f'Intercept: {cafe_lm.intercept_ :.3f}')
print('Coefficients:')

for name,coef in zip(predictors,cafe_lm.coef_):
    print(f' {name}, {coef}')


#Accuracy
fitted = cafe_lm.predict(daily[predictors])

RMSE = np.sqrt(mean_squared_error(daily[outcome],fitted))
r2 = r2_score(daily[outcome],fitted)
print(f'RMSE: {RMSE:.0f}')
print(f'r2: {r2:.4f}')




X = daily[predictors].astype(float)
y = daily[outcome].astype(float)
X = sm.add_constant(X)

model = sm.OLS(y, X)
result = model.fit()

print(result.summary())