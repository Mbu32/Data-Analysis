from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame,Series 

from scipy import stats

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor

import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from pygam import LinearGAM, s, l, te
from pygam.datasets import wage

from dmba import stepwise_selection


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
fitted = housing_lm.predict(housing[predictors])
RMSE = np.sqrt(mean_squared_error(housing[outcome],fitted))
r2 = r2_score(housing[outcome], fitted)
#print(f'RMSE: {RMSE:.0f}')
#print(f'r2: {r2:.4f}')

'''
Initial Linear Regression:

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




#1)Lets compute VIF

features = housing[predictors]
vif = DataFrame()
vif["feature"] = features.columns
vif['VIF'] = [variance_inflation_factor(features.values,i)
              for i in range(features.shape[1])]
#print(vif)
'''
Initial VIF:

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

Turn long/lat into 4 sectors
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
lat_median = housing['Latitude'].median()
long_median = housing['Longitude'].median()
#Separate by north east/west, south east/west
mask_ne = (housing.Longitude > long_median) & (housing.Latitude > lat_median)
mask_nw = (housing.Longitude < long_median) & (housing.Latitude > lat_median)
mask_se = (housing.Longitude > long_median) & (housing.Latitude < lat_median)
mask_sw = (housing.Longitude < long_median) & (housing.Latitude < lat_median)

housing['Region'] = None
housing.loc[mask_ne, 'Region'] = 'NE'
housing.loc[mask_nw, 'Region'] = 'NW'
housing.loc[mask_se, 'Region'] = 'SE'
housing.loc[mask_sw, 'Region'] = 'SW'

region_dummies = pd.get_dummies(housing['Region'], drop_first=True, dtype=int)
housing= pd.concat([housing, region_dummies],axis=1)
housing = housing.drop(columns='Region')

#run our VIF again
predictors = ['MedInc','HouseAge','bdrmsPerRoom',
              'Population','AveOccup',
              'NW','SE','SW'
              ]
outcome = 'MedHouseVal'



features = housing[predictors]
vif_1 = DataFrame()
vif_1["feature_1"] = features.columns
vif_1['VIF_1'] = [variance_inflation_factor(features.values,i)
              for i in range(features.shape[1])]

#print(vif_1)
'''
Second VIF after adjustments made:

   feature_1      VIF_1
0        MedInc   5.535472
1      HouseAge   6.660250
2  bdrmsPerRoom  14.435963
3    Population   2.777088
4      AveOccup   1.095111
5            NW  11.152561
6            SE  12.545445
7            SW   1.601056

Still concerning. Lets try Regularization plus removing SE b/c high multicollinearity with NW (redundant info!!)
'''




predictors_3 = ['MedInc','HouseAge','bdrmsPerRoom',
              'Population','AveOccup',
              'NW','SW'
              ]
outcome = 'MedHouseVal'


LR_ridge = Ridge(alpha=0.5)
LR_ridge.fit(housing[predictors_3],housing[outcome]) 


features = housing[predictors_3]
vif_2 = DataFrame()
vif_2["feature_1"] = features.columns
vif_2['VIF_1'] = [variance_inflation_factor(features.values,i)
              for i in range(features.shape[1])]

#print(vif_2)

'''

Post-Regularization + Engineering of Ratio's

Intercept: -1.431
Coefficients
MedInc : 0.5186842219275254
HouseAge : 0.01650889191126739
bdrmsPerRoom : 4.8287326918273585
Population : 1.4590660735176063e-05
AveOccup : -0.004859689121525546
NW : -0.04713132566994243
SE : 0.29827068826318864



      feature_1     VIF_1
0        MedInc  3.544312
1      HouseAge  6.202707
2  bdrmsPerRoom  7.590573
3    Population  2.728100
4      AveOccup  1.095081
5            NW  1.817965
6            SW  1.062245


Everything is looking much MUCH better/ makes more sense. Less VIF means our estimates are more reliable

Next: Residual Analysis
'''

predictedv = LR_ridge.predict(housing[predictors_3])
residuals = housing[outcome] - predictedv


#print(f"Residuals mean: {residuals.mean():.3f}")  # Should be ~0
#print(f"Residuals std: {residuals.std():.3f}")
dw_stat = durbin_watson(residuals)
#print(f"Durbin-Watson: {dw_stat:.3f}")


'''
Residuals mean: -0.000 ~ amazing
Residuals std: 0.772 
Durbin-Watson: 0.946 ~ a problem, without more data/predictors, this could be hard to fix.

'''




#Outliers
predictors_sm = sm.add_constant(housing[predictors_3]) #we'll be using SM to simplify making the plots... Same answer 
modelridge_sm = sm.OLS(housing[outcome],predictors_sm).fit()
influence = OLSInfluence(modelridge_sm)



'''
MedInc            10.226400
HouseAge          45.000000
AveRooms           3.166667
AveBedrms          0.833333
Population      7460.000000 ~ population of 7.4k but AveOccup of 1.2k?
AveOccup        1243.333333 ~ Cant be right
Latitude          38.320000
Longitude       -121.980000 ~ Searched these up lat/long its on a prison. 
MedHouseVal        1.375000
bdrmsPerRoom       0.263158
NW                 1.000000
SE                 0.000000
SW                 0.000000


Reason for this anamoly: According to google inmates count as population but arent ocnsidered in AveOccup... 
Thats why thos enumbers are skewed.
'''

#lets take a different approach: 

predictors = [
    'MedInc',
    'HouseAge',
    'bdrmsPerRoom',
    'Population',
    'AveOccup',
    'Latitude',
    'Longitude'
]

X = housing[predictors].values
y = housing['MedHouseVal'].values


gam = LinearGAM(
    l(0) +                 # MedInc (linear is fine)
    l(1) +                 # HouseAge
    l(2) +                 # bdrmsPerRoom
    l(3) +                 # Population
    l(4) +                 # AveOccup
    te(5, 6, n_splines=25)  # Latitude, Longitude (2D spatial smooth)
)

gam.gridsearch(X, y)
print(gam.summary())



#residual subplots
fig, ax = plt.subplots(figsize=(10, 8))
fig = sm.graphics.plot_ccpr_grid(modelridge_sm, fig=fig)
plt.tight_layout()
plt.close()


