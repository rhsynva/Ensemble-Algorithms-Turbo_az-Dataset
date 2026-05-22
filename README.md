This is an excellent initiative for your GitHub portfolio. Writing a strong `README.md` that explains not just *what* you did, but *how* and *why* you did it (especially focusing on manual implementations vs. built-in libraries) is exactly what hiring managers and senior data scientists look for.

I have analyzed your `Ensemble_Algorithms_Turbo_az_dataset.ipynb` notebook and the provided text template. Your notebook focuses on predicting car prices (Turbo.az dataset) using various ensemble techniques, including extensive preprocessing, feature engineering, handling data leakage, and comparing Bagging, Boosting, Voting, and Stacking methods.

Here is a professional, detailed `README.md` file tailored for your Turbo.az ensemble project, matching the analytical and descriptive tone of your provided template.


# Turbo.az Car Price Prediction: Advanced Ensemble Techniques

This repository contains my implementation of an end-to-end Machine Learning pipeline focused on predicting car prices using the Azerbaijani automotive classifieds dataset (Turbo.az). Rather than just fitting a single model, this project explores the power of **Ensemble Learning**, culminating in a rigorous comparison of Bagging, Boosting, Voting, and Stacking architectures.

The primary challenge in this dataset involves handling messy, unstructured text features, mitigating severe data leakage, and optimally combining diverse algorithms to achieve a highly accurate regression model.

## Workflow Breakdown

### 1. Data Loading and Exploration

I loaded a dataset consisting of 10,000 car listings. Initial exploratory data analysis revealed significant inconsistencies: mixed currencies (AZN, USD, EUR), string-based engine specifications, and unstructured text in the `Extra` equipment column.

### 2. Data Preprocessing & Currency Standardization

To create a unified target variable, I developed custom extraction functions to parse numeric values from text and standardize all prices into a single `Target_Qiymet_AZN` column based on static exchange rates.

### 3. Feature Engineering (Parsing and Binarization)

Instead of discarding complex text columns, I engineered them into usable numeric features:

* **Engine Parsing:** Split the `Mühərrik` (Engine) column to extract Engine Volume (float), Horsepower (int), and Fuel Type (categorical).
* **Missing Value Imputation (Grouped):** Imputed missing Horsepower and Engine Volume intelligently by grouping the data by `Marka` (Brand) and `Model`, replacing nulls with the median values specific to that exact car model.
* **Text Binarization:** Processed the comma-separated `Extra` column using Pandas string matching to create explicit boolean (1/0) features for specific equipment (e.g., `has_ABS`, `has_Lyuk`, `has_Kamera`).

### 4. Encoding and Outlier Handling

* Applied Target Encoding (Mean Encoding) for high-cardinality features like `Marka` and `Model` to capture price relationships without blowing up the feature space.
* Used One-Hot Encoding for low-cardinality categorical variables.
* Implemented strict Domain Knowledge and Interquartile Range (IQR) filtering to remove anomalies (e.g., extremely low/high prices or mathematically impossible horsepower values), discarding approximately ~5% of noisy data to stabilize the learning process.

### 5. Leakage Resolution and Train/Test Split

During development, a critical data leakage issue was identified where legacy price columns (`Qiymet`, `Təmiz_Qiymət`) remained in the feature set ($X$), causing models to severely overfit (100% $R^2$). I systematically purged all target-correlated columns from the feature matrix before splitting the clean data into an 80% training and 20% testing set.

### 6. Ensemble Architectures (The Core Modeling Logic)

I constructed and evaluated four distinct ensemble paradigms:

1. **Bagging:** `RandomForestRegressor` (Variance reduction via parallel decision trees).
2. **Boosting:** `GradientBoostingRegressor` and `XGBRegressor` (Bias reduction via sequential error correction).
3. **Voting:** `VotingRegressor` (Averaging the predictions of AdaBoost, GradientBoosting, and XGBoost).
4. **Stacking:** `StackingRegressor` (Using a Ridge Regression meta-learner to optimally combine the predictions of the diverse sub-models).

### 7. Evaluation

The models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$ Score. After resolving data leakage and tuning the models, the **Stacking Regressor** emerged as the strongest performer, achieving an $R^2$ score of approximately **92%** and significantly minimizing the MAE, proving that a meta-learner effectively balances the strengths and weaknesses of base boosting algorithms.

## Technologies Used

* **Python 3**
* **Pandas** for complex string manipulation and data wrangling
* **NumPy** for numerical operations
* **Scikit-learn** for base algorithms (RandomForest, GradientBoosting, Ridge, Stacking/Voting) and evaluation metrics
* **XGBoost** for advanced gradient boosting implementations
