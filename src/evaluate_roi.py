import json
import sys
from pathlib import Path

# ----------------------------------
# 1. Read parameters from CLI
# ----------------------------------
threshold = int(sys.argv[1])
bandwidth = int(sys.argv[2])
extra_cost = float(sys.argv[3])

# ----------------------------------
# 2. Load RDD evaluation metrics
# ----------------------------------
rdd_metrics_path = f"metrics/rdd_eval_t{threshold}_bw{bandwidth}.json"

with open(rdd_metrics_path, "r") as f:
    rdd_metrics = json.load(f)

# ----------------------------------
# 3. Compute ROI
# ----------------------------------
recovery_uplift = rdd_metrics["recovery_uplift"]

net_gain = recovery_uplift - extra_cost
roi = net_gain / extra_cost

roi_metrics = {
    "threshold": threshold,
    "bandwidth": bandwidth,
    "extra_recovery_cost": extra_cost,
    "recovery_uplift": recovery_uplift,
    "net_gain": net_gain,
    "roi": roi
}

# ----------------------------------
# 4. Save ROI metrics
# ----------------------------------
Path("metrics").mkdir(exist_ok=True)

output_path = f"metrics/roi_t{threshold}_bw{bandwidth}.json"
with open(output_path, "w") as f:
    json.dump(roi_metrics, f, indent=4)

print(
    f"ROI evaluation completed | "
    f"threshold={threshold}, bandwidth={bandwidth}, ROI={roi:.2f}"
)
