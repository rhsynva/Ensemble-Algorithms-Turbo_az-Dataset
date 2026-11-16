# Turbo.az Car Price Prediction: Advanced Ensemble Techniques

This repository contains an end-to-end machine learning pipeline that predicts car prices from the Azerbaijani automotive classifieds dataset (Turbo.az). Instead of fitting a single model, the project compares four ensemble paradigms (Bagging, Boosting, Voting and Stacking) on a carefully cleaned, leakage-free feature set.

The main challenges in this dataset are messy text features (engine specification, equipment list, mileage), prices in mixed currencies, missing values that are not random, and several sources of target leakage.

## Results

Evaluated once on a held-out test set (80/20 split, `random_state=42`):

| Model | R² | MAE (AZN) | RMSE (AZN) |
|---|---|---|---|
| Stacking Regressor (AdaBoost + GradientBoosting + XGBoost, Ridge meta-learner) | **95.19%** | 3,959 | 8,992 |
| Gradient Boosting | 95.06% | 4,051 | 9,107 |
| Random Forest (Bagging) | 93.65% | 4,432 | 10,327 |
| Voting Regressor (AdaBoost + GradientBoosting + XGBoost) | 91.81% | 8,724 | 11,725 |

Stacking and Gradient Boosting perform best; their 0.13-point gap is within noise for a single split. Voting is weakest, likely because the weak AdaBoost member gets equal weight.

## Workflow Breakdown

1. **Data Loading and Exploration**
   The dataset contains 10,000 car listings. Exploration revealed mixed currencies (7,508 AZN, 2,469 USD, 23 EUR), string-based engine specifications, mileage stored as text (e.g. "370 000 km"), and an unstructured `Extra` equipment column.

2. **Currency Standardization**
   Prices are converted to a single `Target_Qiymet_AZN` column using the `Valyuta` column and fixed exchange rates (1 USD = 1.70 AZN, 1 EUR = 1.85 AZN; the EUR rate is an assumption and affects only 23 listings).

3. **Feature Engineering**
   - **Engine parsing:** `Mühərrik` is split into Engine Volume, Horsepower and Fuel Type.
   - **Grouped imputation:** Missing Horsepower and Engine Volume are filled with the median of the same `Marka` + `Model` group, with a global median as fallback.
   - **Equipment features:** The most frequent items in `Extra` were identified with a frequency count and word cloud, and each of the 13 items became a binary `has_*` feature (e.g. `has_ABS`, `has_Lyuk`, `has_Kamera`), plus `Total_Extras_Count`.
   - **Mileage:** `Yürüş` is parsed into `Yürüş_km`; values above 1M km (28 rows, max 5.8M) are clipped as implausible.
   - **Owner count:** `Sahiblər` is missing in 38% of listings. The hypothesis "missing = new car = 0 owners" was rejected (missing for 72% of new vs 35% of used cars, and 58% of dealer vs 31% of private listings), so missingness is kept as an indicator (`Sahiblər_missing`) and the number is filled with the median.
   - **Other missing values:** `Vəziyyəti` and `Hansı bazar üçün yığılıb` use an explicit "Məlum deyil" category. `Qəzalı` has a single non-empty value ("Bəli", 32 rows), so empty is treated as "not flagged"; flagged cars are much cheaper (median 4,500 vs 22,000 AZN).

4. **Encoding**
   - **Target encoding** for the high-cardinality `Marka` and `Model`, computed on the training split only and applied to the test split with the same mapping.
   - **One-hot encoding** for low-cardinality categorical variables (body type, color, fuel type, market, condition, accident flag, new/used).

5. **Outlier Handling**
   Only implausible records are removed, 34 rows (0.34%) in total:
   - price outside 1,000–400,000 AZN,
   - production year outside 1960–2026,
   - non-motorcycle vehicles with fewer than 30 HP (data-entry errors such as "1 a.g.").

   IQR filtering on horsepower was tested and rejected: it would have removed 486 legitimate premium cars (median price 89,965 AZN vs 20,701 AZN for the rest) and biased the model against expensive vehicles.

6. **Leakage Prevention and Train/Test Split**
   - All price columns (`Qiymet`, `Təmiz_Qiymət`, `Target_Qiymet_AZN`) are excluded from the features.
   - Target encodings are computed after the split, on the training data only.
   - The view count (`Baxışların sayı`) is excluded: it appears only after publication and is itself driven by price.
   - The test set is used only for the final evaluation.

7. **Ensemble Architectures**
   1. **Bagging:** `RandomForestRegressor` (variance reduction via parallel trees).
   2. **Boosting:** `GradientBoostingRegressor` (bias reduction via sequential error correction).
   3. **Voting:** `VotingRegressor` averaging AdaBoost, GradientBoosting and XGBoost.
   4. **Stacking:** `StackingRegressor` with the same three base models and a Ridge Regression meta-learner.

8. **Evaluation**
   Models are compared with MAE, RMSE and R². An ablation (5-fold CV on the training set) showed that the owner-count columns have no measurable benefit (R² 94.61% with vs 94.65% without), since `Yeni`, `Avtosalon` and year likely carry the same signal.

## Problems Found and Fixed During Development

- **Currency bug:** An early version applied the currency detector to an already-numeric price column, which overwrote the original `Valyuta` with "AZN" for every row, so USD prices were never converted. Fixed by using the original `Valyuta` column.
- **Target leakage in encodings:** Brand/model mean-price encodings were first computed on the full dataset before the split. They are now computed on the training split only.
- **Over-aggressive outlier filter:** The IQR rule on horsepower removed ~5% of the data, mostly premium cars. Replaced with domain rules.
- **Incomplete results table:** Only the last model was being stored in the comparison table. Now all four are recorded.

## Limitations

- Single 80/20 split and no hyperparameter tuning; small gaps between models should not be over-interpreted.
- Fixed exchange rates; prices are a snapshot, not time-aware.
- MAE (~4,000 AZN) is much lower than RMSE (~9,000 AZN): errors concentrate on expensive vehicles.
- Text columns `Şəhər`, `Sürətlər qutusu` and `Ötürücü` are not used as features yet.

## Technologies Used

- Python 3
- Pandas for data wrangling and text parsing
- NumPy for numerical operations
- Matplotlib and WordCloud for exploring equipment frequencies
- Scikit-learn for the base algorithms (RandomForest, GradientBoosting, AdaBoost, Ridge, Voting/Stacking), cross-validation and metrics
- XGBoost as a base learner in the Voting and Stacking ensembles
