
# Applied Data Analysis & Statistical Modeling Projects

This repository contains applied projects demonstrating statistical reasoning, regression modeling, feature engineering, and model diagnostics. 
My background is in mathematics, statistics, and finance, and these projects focus on translating theory into practical, data driven insights.



## 1 California Housing Price Modeling

**Objective**  
Model median house values using demographic and geographic predictors, prioritizing **multicollinearity diagnosis, feature engineering, and coefficient interpretability** rather than purely predictive performance.

**Dataset**
- California Housing dataset 
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
- **Advanced Methods**: Generalized Additive Models (GAM) with 2D spatial smoothing
- **Regularization**: Ridge regression for coefficient stabilization
- **Spatial Analysis**: Moran's I test for spatial autocorrelation
- **Residual Diagnostics**: Durbin-Watson, influence points, outlier detection


# Key transformations and analysis steps:
1. Initial VIF analysis revealed VIF > 500 for latitude/longitude
2. Created bdrmsPerRoom = AveBedrms/AveRooms
3. Categorized geographic regions: NE, NW, SE, SW
4. Applied Ridge regression (alpha=0.5) for regularization
5. Built GAM with 2D spatial term: te(Latitude, Longitude, n_splines=25)
6. Conducted Moran's I spatial autocorrelation test (k=8 nearest neighbors)

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

- **Data Manipulation**: pandas, numpy
- **Machine Learning**: scikit-learn (LinearRegression, Ridge, StandardScaler, metrics)
- **Statistical Modeling**: statsmodels (OLS, VIF, Durbin-Watson, influence diagnostics)
- **Scientific Computing**: scipy (statistical functions, optimization)
- **Visualization**: matplotlib, seaborn
- **Advanced Modeling**: pygam (Generalized Additive Models with 2D spatial smoothing)
- **Spatial Analysis**: libpysal (KNN weights), esda (Moran's I spatial autocorrelation)
- **Model Persistence**: pickle (serialization/deserialization of trained models)
- **Model Selection**: dmba (stepwise selection utilities)
