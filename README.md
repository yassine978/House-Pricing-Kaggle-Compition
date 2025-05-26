# 🏡 House Price Prediction - Kaggle Competition

This repository contains my solution to the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/) Kaggle competition. The goal is to predict the sale price of homes in Ames, Iowa using various features provided in the dataset.

---

## 📌 Overview

- **Data Source**: [Kaggle Competition Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- **Model Used**: `CatBoostRegressor` (chosen for its ability to handle categorical variables natively)
- **Performance Metric**: Root Mean Squared Error (RMSE)
- **Environment**: Google Colab + Python + Scikit-Learn + CatBoost

---

## 📂 Project Structure

```
.
├── train.csv                    # Training dataset from Kaggle
├── test.csv                     # Test dataset from Kaggle
├── data_description.txt         # Feature description file
├── house_price_model.ipynb      # Main notebook
├── catboost_bestparam_submission.csv  # Final submission
└── README.md                    # This file
```

---

## 🔍 Exploratory Data Analysis

- **Distribution**: SalePrice is right-skewed; log-transform was applied.
- **Missing Values**:
  - Filled using domain knowledge from `data_description.txt`
  - `LotFrontage`: filled using neighborhood median
  - `MasVnrArea`, `GarageYrBlt`: filled with 0
  - `Electrical`, `KitchenQual`, etc.: filled with mode
- **Visualizations**: Boxplots, histograms, and heatmaps were used to explore relationships between features and target.

---

## 🧠 Feature Engineering

A strong emphasis was placed on generating meaningful features to improve model performance.

### Key Features Created:

- `TotalSF` = `GrLivArea` + `TotalBsmtSF`
- `TotalBath`, `Qual_TotalSF`, `QualityScore`, `GarageScore`
- `IsRemodeled`, `GarageAge`, `Qual_Age`
- `PremiumLocation`, `MSZoning_Quality`, `SummerSale`
- Categorical binning: `MSSubClass_str`
- Boolean flags: `HasGarage`, `HasDeck`, `HasPool`, etc.

All of this was encapsulated in:
```python
def enhanced_feature_engineering(df):
    ...
```

---

## 🧪 Model Training

### Model: CatBoostRegressor

After testing multiple models, CatBoost was selected due to:

- Native handling of categorical variables
- Fast training with GPU acceleration
- Robust performance without heavy tuning

### Final Parameters Used:

```python
CatBoostRegressor(
    learning_rate=0.01,
    l2_leaf_reg=0,
    iterations=3500,
    depth=4,
    colsample_bylevel=0.7,
    border_count=50,
    bootstrap_type='Bayesian',
    bagging_temperature=1,
    random_seed=42
)
```

✅ **Validation RMSE**: *~ your_best_score_here (log scale)*

---

## 📤 Submission Strategy

- Predictions were transformed using `np.expm1()` to reverse log1p.
- Clipping was applied at the 1st and 99th percentiles to remove extreme outliers.
- Output saved to `catboost_bestparam_submission.csv`.

---

## 🛠 Requirements

Install necessary libraries:

```bash
pip install catboost category_encoders scikit-learn matplotlib seaborn
```

---

## 👤 Author

**Madhi**  
Master's student in **Big Data & Artificial Intelligence**  
Paris-Dauphine | Tunis  
🇹🇳

---

## 📚 Acknowledgments

- [Kaggle - House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/)
- Feature descriptions from `data_description.txt`
- CatBoost documentation and public kernels on Kaggle

---
