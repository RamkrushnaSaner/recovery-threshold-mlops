import pandas as pd
import statsmodels.api as sm
import json
import pickle
import numpy as np

# Load processed data
df = pd.read_csv("data/processed/bank_data.csv")

# Feature engineering
df["indicator_1000"] = (df["expected_recovery_amount"] >= 1000).astype(int)

# Define X and y
X = df[["expected_recovery_amount", "indicator_1000"]]
y = df["actual_recovery_amount"]

X = sm.add_constant(X)

# Train OLS model
model = sm.OLS(y, X).fit()

# Save model
with open("models/ols_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Predictions
y_pred = model.predict(X)

# Error metrics
rmse = np.sqrt(np.mean((y - y_pred) ** 2))
mae = np.mean(np.abs(y - y_pred))

# Confidence interval for indicator
conf_int = model.conf_int().loc["indicator_1000"]

# Metrics dictionary
metrics = {
    # Model quality
    "r_squared": model.rsquared,
    "adj_r_squared": model.rsquared_adj,
    "aic": model.aic,
    "bic": model.bic,

    # Statistical validity
    "p_value_indicator_1000": model.pvalues["indicator_1000"],
    "t_stat_indicator_1000": model.tvalues["indicator_1000"],
    "f_stat_pvalue": model.f_pvalue,

    # Business interpretation
    "coef_indicator_1000": model.params["indicator_1000"],
    "coef_ci_lower_95": conf_int[0],
    "coef_ci_upper_95": conf_int[1],

    # Error metrics
    "rmse": rmse,
    "mae": mae
}

# Save metrics
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("OLS training completed successfully")
