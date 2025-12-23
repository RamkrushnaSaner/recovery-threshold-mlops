import pandas as pd
import statsmodels.api as sm
import json
import pickle
import numpy as np
import sys
from pathlib import Path

# -----------------------------
# 1. Read threshold from CLI
# -----------------------------
# Example: python src/train.py 1000
threshold = int(sys.argv[1])

# -----------------------------
# 2. Load processed data
# -----------------------------
DATA_PATH = "data/processed/bank_data.csv"
df = pd.read_csv(DATA_PATH)

# -----------------------------
# 3. Feature engineering
# -----------------------------
# Indicator for threshold crossing
df["indicator"] = (df["expected_recovery_amount"] >= threshold).astype(int)

# -----------------------------
# 4. Define X and y
# -----------------------------
X = df[["expected_recovery_amount", "indicator"]]
y = df["actual_recovery_amount"]

X = sm.add_constant(X)

# -----------------------------
# 5. Train OLS model
# -----------------------------
model = sm.OLS(y, X).fit()

# -----------------------------
# 6. Save model artifact
# -----------------------------
Path("models").mkdir(exist_ok=True)

with open("models/ols_model.pkl", "wb") as f:
    pickle.dump(model, f)

# -----------------------------
# 7. Predictions & error metrics
# -----------------------------
y_pred = model.predict(X)

rmse = np.sqrt(np.mean((y - y_pred) ** 2))
mae = np.mean(np.abs(y - y_pred))

# Confidence interval for indicator
conf_int = model.conf_int().loc["indicator"]

# -----------------------------
# 8. Collect metrics
# -----------------------------
metrics = {
    # Experiment parameter
    "threshold": threshold,

    # Model quality
    "r_squared": model.rsquared,
    "adj_r_squared": model.rsquared_adj,
    "aic": model.aic,
    "bic": model.bic,

    # Threshold effect (business + stats)
    "coef_indicator": model.params["indicator"],
    "coef_ci_lower_95": conf_int[0],
    "coef_ci_upper_95": conf_int[1],
    "p_value_indicator": model.pvalues["indicator"],
    "t_stat_indicator": model.tvalues["indicator"],

    # Overall model validity
    "f_stat_pvalue": model.f_pvalue,

    # Error metrics
    "rmse": rmse,
    "mae": mae
}

# -----------------------------
# 9. Save metrics
# -----------------------------
Path("metrics").mkdir(exist_ok=True)

with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"OLS training completed successfully for threshold = {threshold}")
