import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from scipy import stats
import statsmodels.api as sm

# =========================================================
# 1. Read parameters from CLI
# =========================================================
# Example:
# python src/evaluate_rdd.py 1000 100
# threshold = 1000, bandwidth = ±100
threshold = int(sys.argv[1])
bandwidth = int(sys.argv[2])

# =========================================================
# 2. Load processed data
# =========================================================
DATA_PATH = "data/processed/bank_data.csv"
df = pd.read_csv(DATA_PATH)

# =========================================================
# 3. Create RDD window (local neighborhood)
# =========================================================
lower = threshold - bandwidth
upper = threshold + bandwidth

rdd_df = df.loc[
    (df["expected_recovery_amount"] >= lower) &
    (df["expected_recovery_amount"] < upper)
].copy()

# Indicator for policy
rdd_df["indicator"] = (rdd_df["expected_recovery_amount"] >= threshold).astype(int)

# Split groups
control = rdd_df[rdd_df["indicator"] == 0]
treatment = rdd_df[rdd_df["indicator"] == 1]

# =========================================================
# 4. Balance checks (as in original notebook)
# =========================================================
# Age balance
kw_age_stat, kw_age_p = stats.kruskal(
    control["age"],
    treatment["age"]
)

# Sex balance (chi-square)
crosstab_sex = pd.crosstab(
    rdd_df["indicator"],
    rdd_df["sex"]
)

chi2_sex_stat, chi2_sex_p, _, _ = stats.chi2_contingency(crosstab_sex)

# =========================================================
# 5. Recovery comparison (non-parametric)
# =========================================================
kw_recovery_stat, kw_recovery_p = stats.kruskal(
    control["actual_recovery_amount"],
    treatment["actual_recovery_amount"]
)

# Mean recovery difference (business impact)
mean_recovery_control = control["actual_recovery_amount"].mean()
mean_recovery_treatment = treatment["actual_recovery_amount"].mean()
recovery_uplift = mean_recovery_treatment - mean_recovery_control

# =========================================================
# 6. Local linear RDD regression
# =========================================================
X = rdd_df[["expected_recovery_amount", "indicator"]]
y = rdd_df["actual_recovery_amount"]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# =========================================================
# 7. Collect evaluation metrics
# =========================================================
metrics = {
    # Experiment parameters
    "threshold": threshold,
    "bandwidth": bandwidth,
    "n_samples": len(rdd_df),

    # Balance checks
    "kw_age_pvalue": kw_age_p,
    "chi2_sex_pvalue": chi2_sex_p,

    # Recovery effect
    "mean_recovery_control": mean_recovery_control,
    "mean_recovery_treatment": mean_recovery_treatment,
    "recovery_uplift": recovery_uplift,
    "kw_recovery_pvalue": kw_recovery_p,

    # Local RDD regression
    "coef_indicator": model.params["indicator"],
    "p_value_indicator": model.pvalues["indicator"],
    "ci_lower_95": model.conf_int().loc["indicator"][0],
    "ci_upper_95": model.conf_int().loc["indicator"][1],
    "r_squared": model.rsquared
}

# =========================================================
# 8. Save metrics
# =========================================================
Path("metrics").mkdir(exist_ok=True)

output_path = f"metrics/rdd_eval_t{threshold}_bw{bandwidth}.json"
with open(output_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(
    f"RDD evaluation completed | threshold={threshold}, bandwidth={bandwidth}"
)
