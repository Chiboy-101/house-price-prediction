# 🏠 House Price Prediction with Gradient Boosting

A machine learning project that predicts house prices using advanced feature engineering and ensemble models.  
The final tuned **Gradient Boosting Regressor** achieved strong performance:

> **R² = 0.917**  
> **RMSE = 93,783.77**

This project demonstrates an end-to-end ML workflow: data preprocessing, feature engineering, model comparison, hyperparameter tuning, and model deployment preparation.

---

## 📌 Project Overview

House prices depend on many factors such as location, size, age of the building, and renovation history.  
This project builds a regression model to predict house prices using the **King County Housing Dataset**.

The goal was to minimize prediction error (RMSE) while maintaining high explanatory power (R²).

---

## ✨ Key Features

- Data cleaning and preprocessing  
- Feature engineering (age, renovation features, area ratios, interactions)  
- Categorical encoding (zipcode)  
- Outlier handling (price capped at 99th percentile)  
- Model comparison:
  - Linear Regression  
  - Random Forest Regressor  
  - Gradient Boosting Regressor (best model)  
- Hyperparameter tuning with RandomizedSearchCV  
- Model persistence using Pickle  

---

## 🧠 Models & Performance

| Model                     | R²     | RMSE        |
|---------------------------|--------|-------------|
| Linear Regression         | 0.8529 | 124, 683.32  |
| Random Forest Regressor   | 0.8877 | 108, 915.85  |
| Gradient Boosting   | 0.8957 | 104, 985.54 |


---

## 🚀 How to Run the Project

 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction

2️⃣ Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # macOS/Linux

3️⃣ Install dependencies
```bash
pip install -r requirements.txt

4️⃣ Train the model
```bash
python src/train.py
