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
import statsmodels.stats.api as sms
from scipy.stats import bootstrap





sales = pd.read_csv('dirty_cafe_sales.csv')

#print(sales.info())

cdata = sales.copy()

#Converting numeric columns to numeric data type
cdata['Quantity'] = pd.to_numeric(cdata['Quantity'],errors='coerce').astype('Int64')  #adding the .astype('Int64') removes decimals, capital I in Int allows us to keep our NA values
cdata['Price Per Unit'] = pd.to_numeric(cdata['Price Per Unit'],errors='coerce')
cdata['Total Spent'] = pd.to_numeric(cdata['Total Spent'],errors='coerce')


#date columns into the date data type
cdata['Transaction Date'] = pd.to_datetime(cdata['Transaction Date'],errors='coerce')


#Convert object columns to the category data type
cdata['Item'] = cdata['Item'].astype('category')  #stores each variable as a code now, saves memory
cdata['Payment Method'] = cdata['Payment Method'].astype('category')
cdata['Location'] = cdata['Location'].astype('category')



#print(cdata.info())




#histograms for numerical bar charts for categorical

fig, axes= plt.subplots(2,3, figsize=(15,15))  
fig.suptitle('Summary Visualization for clean data Columns', fontsize = 14)



# Numeric Columns: Histograms
numeric_cols = ['Quantity','Price Per Unit', 'Total Spent']

for i,col in enumerate(numeric_cols):
    sns.histplot(data=cdata[col].dropna(),ax=axes[0][i], bins=15)
    axes[0][i].set_title(f"Distribution of {col}")
    axes[0][i].set_xlabel(col)
    axes[0][i].set_ylabel('Count')

#Categorical Columns: Bar plots for value counts
categorical_cols = ['Item','Payment Method','Location']
for i,col in enumerate(categorical_cols):
    col_str = cdata[col].astype(str).replace(['nan','UNKNOWN','ERROR'],'NaN')
    sns.countplot(y=col_str, order=col_str.value_counts().index , ax=axes[1][i])
    axes[1][i].set_title(f"Frequency of {col}")
    axes[1][i].set_xlabel('Count')
    axes[1][i].set_ylabel(col)


plt.tight_layout(rect=[0,0,1,.96])


#before capping
print(cdata['Total Spent'].quantile([0,.05,.1,.25,.5,.75,.9,.95,.99,1]))

#Dont really need to cap since numbers made sense, but lets do 95th percentile for practice
cap_value= cdata['Total Spent'].quantile(.95)
min_value = cdata['Total Spent'].quantile(.05)
cdata.loc[cdata['Total Spent']> cap_value,'Total Spent'] = cap_value 
cdata.loc[cdata['Total Spent']<min_value,'Total Spent'] = min_value
#After capping
print('This is after capping the data to the 95ht percentile \n',cdata['Total Spent'].quantile([0,.05,.1,.25,.5,.75,.9,.95,.99,1]))

plt.close()

#print(f"This is the median: {cdata['Total Spent'].median()}",f"While this is the mean {cdata['Total Spent'].mean()}")









# H0 = Mean of $ spent on Takeout equal to Mean of In-Store
#Alpha =0.05

how1 = cdata.loc[cdata['Location'].isin(['In-store','Takeaway']),['Location','Total Spent']].dropna()
how = how1.loc[how1['Location'].isin(['In-store','Takeaway']),'Location'].value_counts()
print(how)




takeaway_spent= how1.loc[how1['Location'] == 'Takeaway','Total Spent']
instore_spent= how1.loc[how1['Location'] == 'In-store','Total Spent']


res = stats.ttest_ind(takeaway_spent,instore_spent, equal_var=False)

alpha = 0.05
if alpha > res.pvalue:
    print(f"Reject Null-Hypothesis with a p value of {res.pvalue :.4f}") #p<alpha
else:
    print(f"Fail to reject Null-Hypothesis with a p value of {res.pvalue :.4f}") #p>alpha



#CONFIDENCE INTERVAL

takeaway_stats = sms.DescrStatsW(takeaway_spent)
instore_stats = sms.DescrStatsW(instore_spent)

ci = sms.CompareMeans(takeaway_stats, instore_stats).tconfint_diff()
print(ci)




