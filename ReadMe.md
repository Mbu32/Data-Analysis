
## Overview

The project analyzes cafe sales data through two main components:

### 1. Data Cleaning and Hypothesis Testing (`CleanCafe.py`)
- Cleans and preprocesses raw sales data
- Performs exploratory data analysis with visualizations
- Conducts statistical hypothesis testing comparing spending between in-store and takeaway customers

### 2. Revenue Forecasting (`MLRCafe.py`)
- Aggregates transaction-level data into daily revenue metrics
- Builds linear regression models for short-horizon revenue forecasting
- Performs outlier detection and model diagnostics

## Key Features

### Part 1: Data Cleaning and Hypothesis Testing (`CleanCafe.py`)
**Data Cleaning:**
- Converts data types for numeric, date, and categorical columns
- Handles missing values appropriately
- Caps outliers in the 'Total Spent' column using the 95th percentile

**Visualizations:**
- Histograms for numerical variables (Quantity, Price Per Unit, Total Spent)
- Bar charts for categorical variables (Item, Payment Method, Location)
- Multi-panel visualization layout for comprehensive data overview

**Statistical Analysis:**
- Independent t-test comparing mean spending between in-store and takeaway customers
- Calculation of confidence intervals for the difference in means
- Hypothesis testing with α = 0.05 significance level

### Part 2: Revenue Forecasting (`MLRCafe.py`)
**Data Aggregation:**
- Transforms transaction-level data into daily aggregates
- Creates features: DailyRevenue, TotalQuantity, DayOfWeek, IsWeekend

**Regression Modeling:**
- Linear regression model with DailyRevenue as outcome
- Predictors: DayOfWeek, IsWeekend, TotalQuantity
- Both scikit-learn and statsmodels implementations

**Model Diagnostics:**
- RMSE and R² evaluation metrics
- Outlier detection using studentized residuals
- Visualization of residuals vs. fitted values

