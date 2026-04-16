# Flood Risk Prediction: Capstone Project

## 🌊 Project Overview
This project focuses on predicting flood probability across **1.1 million geographic locations**. It was developed as a **Capstone Project** to demonstrate proficiency in handling large-scale datasets, advanced feature engineering, and ensemble machine learning techniques.

The core challenge was to transform 20 environmental and social indicators into a high-precision risk score using Gradient Boosting algorithms.

## 📊 Key Results
* **Champion Model:** XGBoost Regressor
* **Performance Metric ($R^2$):** **0.8690**
* **Mean Absolute Error (MAE):** 0.0143
* **Dataset Size:** 1,117,957 Rows

---

## 🛠️ Implementation Details

### 1. Feature Engineering
Raw data alone provided a limited baseline. I engineered **10 statistical meta-features** to capture the "aggregate risk profile" of each region, which significantly improved the model's predictive power:
* **Central Tendency:** `sum`, `mean`, `median`
* **Dispersion:** `std`, `cov`, `range`
* **Distribution Extremes:** `p25`, `p75`, `max`, `min`

### 2. Model Selection & Comparison
I conducted a comparative analysis of three distinct algorithms to find the optimal balance between speed and precision:

| Model | Use Case | Key Strength |
| :--- | :--- | :--- |
| **ElasticNet** | Baseline | Robust regularization ($L1$/$L2$) for linear trends. |
| **LightGBM** | Scaling | Leaf-wise growth for fast processing of 1M+ rows. |
| **XGBoost** | **Final Model** | Level-wise growth; best at capturing complex non-linearities. |

### 3. Hyperparameter Tuning
The final XGBoost model was optimized using `GridSearchCV` with the following configuration:
- `learning_rate`: 0.07
- `max_depth`: 10
- `n_estimators`: 300
- `subsample`: 0.9
- `min_child_weight`: 3

---

## 🚀 Usage Instructions

### **1. Prerequisites**
Install the necessary data science stack:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
