from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame,Series 



from scipy import stats
from scipy.stats import bootstrap

from sklearn.utils import resample
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression


import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.outliers_influence import variance_inflation_factor


from pygam import LinearGAM, s, l
from pygam.datasets import wage

from dmba import stepwise_selection
from dmba import AIC_score


'''
California Housing dataset
--------------------------

**Data Set Characteristics:**

:Number of Instances: 20640

:Number of Attributes: 8 numeric, predictive attributes and the target

:Attribute Information:
    - MedInc        median income in block group
    - HouseAge      median house age in block group
    - AveRooms      average number of rooms per household
    - AveBedrms     average number of bedrooms per household
    - Population    block group population
    - AveOccup      average number of household members
    - Latitude      block group latitude
    - Longitude     block group longitude

:Missing Attribute Values: None

This dataset was obtained from the StatLib repository.
https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

The target variable is the median house value for California districts,
expressed in hundreds of thousands of dollars ($100,000).

This dataset was derived from the 1990 U.S. census, using one row per census
block group. A block group is the smallest geographical unit for which the U.S.
Census Bureau publishes sample data (a block group typically has a population
of 600 to 3,000 people).

A household is a group of people residing within a home. Since the average
number of rooms and bedrooms in this dataset are provided per household, these
columns may take surprisingly large values for block groups with few households
and many empty houses, such as vacation resorts.

It can be downloaded/loaded using the
:func:`sklearn.datasets.fetch_california_housing` function.

.. rubric:: References

- Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
  Statistics and Probability Letters, 33:291-297, 1997.

'''




house = fetch_california_housing(as_frame=True)
housing = house.frame
#print(housing.head())
#print(housing.columns)
#print(housing.info())

predictors = ['MedInc','HouseAge','AveRooms','AveBedrms',
              'Population','AveOccup',
              'Latitude','Longitude']
outcome = 'MedHouseVal'





housing_lm = LinearRegression()
housing_lm.fit(housing[predictors],housing[outcome])

print(f'Intercept: {housing_lm.intercept_:.3f}')
print('Coefficients')

for name, coef in zip(predictors, housing_lm.coef_):
    print(f'{name} : {coef}')

fitted = housing_lm.predict(housing[predictors])
RMSE = np.sqrt(mean_squared_error(housing[outcome],fitted))
r2 = r2_score(housing[outcome], fitted)
print(f'RMSE: {RMSE:.0f}')
print(f'r2: {r2:.4f}')

'''
Linear regression:
Intercept: -36.942
Coefficients
MedInc : 0.4366932931343252
HouseAge : 0.009435778033238536
AveRooms : -0.10732204139090426
AveBedrms : 0.6450656935198138
Population : -3.9763894211976986e-06
AveOccup : -0.003786542654970875
Latitude : -0.42131437752714324
Longitude : -0.4345137546747779

RMSE: 1
r2: 0.6062
'''


#Checking Correlation among predictors
#Interactions
#Partial Residuals for nonlinearity & interactions
#Variance distribution/patter

#1)Lets compute VIF

features = housing[predictors]
vif = DataFrame()
vif["feature"] = features.columns
vif['VIF'] = [variance_inflation_factor(features.values,i)
              for i in range(features.shape[1])]

print(vif)



'''
 feature         VIF
0      MedInc   11.511140
1    HouseAge    7.195917
2    AveRooms   45.993601
3   AveBedrms   43.590314
4  Population    2.935745
5    AveOccup    1.095243
6    Latitude  559.874071
7   Longitude  633.711654

Unstable coefficients: 6/7 & 2/3

2/3 ~ AveRooms definitely includes Bedrooms
6/7 ~ they are encoding location

Fixes:
make Avebedrooms as a ratio of Averooms.
Averooms_fixed = AveBedrms/AveRooms 
~This wont tell us exact amount of rooms, but what fraction of rooms are bedrooms!

'''


#Fix bdrms/rooms
housing['bdrmsPerRoom'] = housing['AveBedrms']/housing['AveRooms']


#making sure they all fall between (0,1)
counts = {
    'Below 0': (housing['bdrmsPerRoom'] < 0).sum(),
    '0 to 1': ((housing['bdrmsPerRoom'] >= 0) & (housing['bdrmsPerRoom'] <= 1)).sum(),
    'Above 1': (housing['bdrmsPerRoom'] > 1).sum()
}
print(counts)


##now for long/lat
#print(housing.Longitude,housing.Latitude)
lat_median = housing['Latitude'].median()
long_median = housing['Longitude'].median()

housing['NE'] = (housing.Longitude > long_median) & housing.Latitude.loc[-122.23:]



#run our VIF again
predictors = ['MedInc','HouseAge','bdrmsPerRoom',
              'Population','AveOccup',
              'Latitude','Longitude']
outcome = 'MedHouseVal'

features = housing[predictors]
vif_1 = DataFrame()
vif_1["feature_1"] = features.columns
vif_1['VIF_1'] = [variance_inflation_factor(features.values,i)
              for i in range(features.shape[1])]

print(vif_1)



















#Influence & outliers

house_outliers = sm.OLS(housing[outcome],housing[predictors].assign(const=1))
outliers_result = house_outliers.fit()
#print(outliers_result.summary())
#corr_matrix = housing[predictors].corr()
#print(corr_matrix)




'''
influence = OLSInfluence(house_outliers.fit())
fig, ax = plt.subplots(figsize=(5,5))
ax.axhline(-2.5,linestyle='--',color='C1')
ax.axhline(2.5,linestyle='--',color='C1')
ax.scatter(influence.hat_matrix_diag,influence.resid_studentized_internal,
           s=1000 * np.sqrt(influence.cooks_distance[0]),alpha=0.5)

ax.set_xlabel('hat values')
ax.set_ylabel('studentized residuals')

plt.tight_layout()
plt.show()
'''




'''
mask = [dist<0.8 for dist in influence.cooks_distance[0]]
house_infl = house_outliers.loc[mask]
ols_infl = sm.OLS(house_infl[outcome], house_infl[predictors])
result_infl = ols_infl.fit()
pd.DataFrame({
    'Original': housing_lm.params,
    'Influential removed': result_infl.params,
})
'''
