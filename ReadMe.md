# Applied Data Analysis & Statistical Modeling Projects

This repository contains applied projects demonstrating **statistical reasoning, feature engineering, model diagnostics, and supervised learning** in real-world datasets.  
Drawing on my background in mathematics, statistics, and finance, these projects apply Python and modern machine learning techniques with an emphasis on interpretability and sound methodology.

---

## 1. Credit Card Default Risk Modeling

### Objective
Develop and compare classification models for **credit default prediction**, with emphasis on:
- Proper handling of class imbalance
- Feature diagnostics and multicollinearity control
- Threshold optimization for decision-making
- Out-of-sample validation
- Interpretability in a risk management context

### Dataset
- Taiwanese Credit Card Default Dataset  
- ~30,000 observations  
- Binary target: `default payment next month`

### Data Preparation & Feature Engineering
- Categorical encoding:
  - Age binned into risk-relevant intervals
  - Education and marital status consolidated and dummy encoded
- Feature selection guided by:
  - Variance Inflation Factor (VIF)
  - Domain knowledge
- Linearity checks of predictors with respect to the log-odds using Box–Tidwell-style diagnostics
- Continuous and categorical features handled separately using `ColumnTransformer`

### Models Implemented

#### K-Nearest Neighbors (Exploratory)
- Hyperparameter tuning via GridSearchCV
- Probability-based prediction using cross-validated out-of-fold estimates
- Threshold optimization for F1-score
- Ultimately excluded due to weak generalization and instability

#### Logistic Regression (Primary Baseline)
- Standardized continuous variables
- Class imbalance handled
- Cross-validated probability estimates
- Explicit **decision threshold optimization**
- Evaluation via confusion matrix, precision, recall, specificity, F1-score, and ROC-AUC

#### Generalized Additive Model (Logistic GAM)
- Smooth terms for nonlinear financial predictors
- Factor terms for categorical variables
- Threshold tuning for classification
- Was unable to complete afull smoothing parameter grid search due to memory constraints 

#### Random Forest Classifier
- Out-of-bag (OOB) error estimation
- Permutation-based feature importance
- Feature ablation experiments showing dominance of repayment status (`PAY_0`)
- Comparable accuracy achieved with dramatically reduced feature sets

### Key Findings
- Repayment history (`PAY_0`) dominates predictive performance
- Most demographic variables contributed marginally or simply added noise
- Threshold tuning materially improves F1 without retraining models
- Oversampling techniques (SMOTE variants) increased overfitting and were excluded
- Simpler, interpretable models performed competitively with complex ensembles

---

## 2. California Housing Price Modeling

### Objective
Model median house values using demographic and geographic predictors, prioritizing **multicollinearity diagnosis, feature engineering, spatial structure, and coefficient interpretability**.

### Dataset
- California Housing dataset  
- 20,640 observations, 8 numerical predictors  
- Target: Median house value (in \$100,000s)

### Methods & Techniques
- Linear regression (scikit-learn, statsmodels)
- Model evaluation using RMSE and R²
- Multicollinearity diagnosis via **Variance Inflation Factor (VIF)**
- Feature engineering:
  - Bedroom-to-room ratio (`AveBedrms / AveRooms`)
  - Region-based geographic encoding using dummy variables
- Model re-estimation after feature refinement
- Residual and influence diagnostics using **statsmodels**
- **Regularization**: Ridge regression for coefficient stabilization
- **Generalized Additive Models (GAM)** with 2D spatial smoothing
- **Spatial Analysis**: Moran’s I test for spatial autocorrelation

### Key Transformations & Analysis Steps
1. Initial VIF analysis revealed severe multicollinearity (VIF > 500) for latitude/longitude  
2. Engineered `bdrmsPerRoom` to stabilize housing structure effects  
3. Categorized geographic regions (NE, NW, SE, SW)  
4. Applied Ridge regression (α = 0.5)  
5. Built GAM with spatial smooth: `te(Latitude, Longitude, n_splines=25)`  
6. Conducted Moran’s I test using KNN spatial weights (k = 8)

---

## Tools & Libraries

### Core Stack
- **Data Manipulation**: pandas, numpy  
- **Visualization**: matplotlib, seaborn  
- **Machine Learning**: scikit-learn  
- **Statistical Modeling**: statsmodels  
- **Generalized Additive Models**: pygam  

### Modeling & Validation
- Logistic Regression, KNN, Random Forest, Decision Trees
- Cross-validation, GridSearchCV, out-of-fold prediction
- Decision threshold optimization
- Permutation-based feature importance

### Additional Tools
- imbalanced-learn (SMOTE, ADASYN, BorderlineSMOTE)
- xgboost (XGBClassifier)
- dmba (decision tree visualization utilities)
- pickle (model and artifact persistence)

---

## Project Philosophy

These projects emphasize:
- **Interpretability and diagnostics**
- **Clear separation of training, validation, and testing**
- **Decision-aware evaluation (thresholds, costs, and trade-offs)**

They are designed to reflect how models are evaluated and deployed in risk management, finance, and applied data science.
