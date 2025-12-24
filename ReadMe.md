
# Applied Data Analysis & Statistical Modeling Projects

This repository contains applied projects demonstrating **statistical reasoning, regression modeling, feature engineering, and model diagnostics**, with an emphasis on interpretability and sound quantitative decision making.  
My background is in **mathematics, statistics, and finance**, and these projects focus on translating theory into practical, data driven insights.



## 1 California Housing Price Modeling

**Objective**  
Model median house values using demographic and geographic predictors, prioritizing **multicollinearity diagnosis, feature engineering, and coefficient interpretability** rather than purely predictive performance.

**Dataset**
- California Housing dataset (1990 U.S. Census)
- 20,640 observations, 8 numerical predictors  
- Target: Median house value (in \$100,000s)

**Methods & Techniques**
- Linear regression (scikitlearn)
- Model evaluation using RMSE and R²
- Multicollinearity diagnosis via **Variance Inflation Factor (VIF)**
- Feature engineering:
  - Bedroom to room ratio (`AveBedrms / AveRooms`)
  - Region based geographic encoding using dummy variables
- Model re estimation after feature refinement
- Residual and influence diagnostics using **statsmodels**

**Key Insights**
- Severe multicollinearity was identified among room related and geographic predictors
- Feature engineering substantially improved coefficient stability and interpretability
- Reduced model complexity led to an expected decrease in R², reflecting a trade off between fit and interpretability

**Skills Demonstrated**
- Statistical diagnostics  
- Feature engineering  
- Regression interpretation  
- Bias variance trade offs  

---

##  Café Sales Analysis & Revenue Forecasting

**Objective**  
Analyze transactional café data to understand customer spending behavior and build **short horizon revenue forecasts** using regression based methods.

---

### Part A: Data Cleaning & Hypothesis Testing (`CleanCafe.py`)

**Data Preparation**
- Data type conversions for numeric, date, and categorical variables
- Missing value handling
- Outlier treatment via 95th percentile capping on total spend

**Exploratory Data Analysis**
- Histograms for numerical variables (quantity, price per unit, total spent)
- Bar charts for categorical variables (item, payment method, location)

**Statistical Analysis**
- Independent two sample t-test comparing in store vs takeaway spending
- Confidence interval estimation for the difference in means
- Hypothesis testing at significance level α = 0.05

---

### Part B: Revenue Forecasting (`MLRCafe.py`)

**Feature Engineering**
- Daily aggregation of transaction level data
- Construction of:
  - Daily revenue
  - Total quantity sold
  - Day of week indicators
  - Weekend indicator

**Modeling**
- Linear regression implemented with both scikitlearn and statsmodels
- Model evaluation using RMSE and R²

**Diagnostics**
- Residuals vs fitted values visualization
- Outlier detection using studentized residuals
- Model validity checks

---

## Tools & Libraries

- Python  
- pandas, numpy  
- scikitlearn  
- statsmodels  
- scipy  
- matplotlib, seaborn  

