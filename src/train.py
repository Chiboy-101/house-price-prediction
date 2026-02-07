import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle

# Load the dataset
df = pd.read_csv("kc_house_data.csv")

# Drop unecessary columns
df.drop("id", axis=1, inplace=True)

# Make sure 'date' is in datetime format
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Encode zipcode as categorical variable
df = pd.get_dummies(df, columns=["zipcode"], drop_first=True)

# Extract year and month of sale from date
df["sale_year"] = df["date"].dt.year
df["sale_month"] = df["date"].dt.month

# House age (age at time of sale)
df["age"] = df["sale_year"] - df["yr_built"]

# Whether the house was ever renovated (1 if yes, 0 if no)
df["was_renovation"] = (df["yr_renovated"] > 0).astype(int)

# Time since renovation (if never renovated, set to 0)
df["years_since_renovation"] = df["sale_year"] - df["yr_renovated"]
df.loc[df["yr_renovated"] == 0, "years_since_renovation"] = 0

# Living area per floor (avoid division by zero)
df["living_area_per_floor"] = df["sqft_living"] / df["floors"].replace(0, 1)

# Lot area per bedroom (avoid division by zero)
df["lot_per_bedroom"] = df["sqft_lot"] / df["bedrooms"].replace(0, 1)

# Basement proportion (part of total living space)
df["basement_ratio"] = df["sqft_basement"] / df["sqft_living"].replace(0, 1)

# Combined or derived features
df["total_area"] = df["sqft_living"] + df["sqft_basement"]
df["bath_per_bed"] = df["bathrooms"] / df["bedrooms"].replace(0, 1)

# Interaction features
df["bed_bath_interaction"] = df["bedrooms"] * df["bathrooms"]
df["living_per_lot"] = df["sqft_living"] / df["sqft_lot"]
df["bed_bath_ratio"] = df["bathrooms"] / df["bedrooms"].replace(0, 1)
df["living_lot_ratio"] = df["sqft_living"] / df["sqft_lot"]
df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

# drop the original date
df.drop(columns=["date"], inplace=True)

# Explore and vizualize the dataset
df.info()

# cap extreme prices to the 99th percentile
q_high = df["price"].quantile(0.99)
df["price"] = np.where(df["price"] > q_high, q_high, df["price"])

# Now check and very if outliers are gone
df.boxplot("price")
plt.show()

# Select features and target variable
X = df.drop("price", axis=1)
y = df["price"]

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale all numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Use a for loop to see with models are best
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(
        n_estimators=200, random_state=42
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name} -> R²: {r2:.4f}, RMSE: {rmse:.2f}")

# Define parameter grid
param_grid = {
    "n_estimators": [300, 500, 700, 1000],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "subsample": [0.8, 0.9, 1.0],
    "max_features": ["sqrt", "log2", None],
}

# Initialize base model
gbr = GradientBoostingRegressor(random_state=42)

# Randomized search
gbr_cv = RandomizedSearchCV(
    estimator=gbr,
    param_distributions=param_grid,
    n_iter=30,  # number of random combinations
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,
    verbose=2,
    scoring="neg_root_mean_squared_error",  # optimize RMSE directly
    random_state=42,
)

# Fit on training data
gbr_cv.fit(X_train, y_train)

# Best hyperparameters and score
print("Best params:", gbr_cv.best_params_)
print("Best RMSE :", -gbr_cv.best_score_)

# Evaluate on test set
best_gbr = gbr_cv.best_estimator_
y_pred = best_gbr.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print the results
print(f"Tuned R²: {r2:.3f}")
print(f"Tuned RMSE: {rmse:.2f}")

# Save the model
with open("best_gbr_model.pkl", "wb") as f:
    pickle.dump(best_gbr, f)
