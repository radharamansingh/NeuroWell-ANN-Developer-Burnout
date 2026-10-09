
import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible synthetic data
np.random.seed(42)

N = 10000

# Developer workload and wellbeing indicators
df = pd.DataFrame({
    "working_hours": np.random.uniform(4, 14, N).round(1),
    "sleep_hours": np.random.uniform(3.5, 9, N).round(1),
    "coding_hours": np.random.uniform(1, 10, N).round(1),
    "meetings_per_day": np.random.randint(0, 9, N),
    "breaks_per_day": np.random.randint(0, 8, N),
    "deadline_pressure": np.random.randint(1, 11, N),
    "job_satisfaction": np.random.randint(1, 11, N),
    "work_life_balance": np.random.randint(1, 11, N),
    "consecutive_working_days": np.random.randint(1, 15, N),
    "tasks_completed": np.random.randint(1, 16, N)
})

# Simulated risk score for creating teaching labels.
# These rules are illustrative, not clinically validated.
risk_score = (
    2.0 * df["working_hours"]
    - 1.8 * df["sleep_hours"]
    + 0.8 * df["coding_hours"]
    + 0.8 * df["meetings_per_day"]
    - 0.7 * df["breaks_per_day"]
    + 1.5 * df["deadline_pressure"]
    - 1.2 * df["job_satisfaction"]
    - 1.0 * df["work_life_balance"]
    + 0.7 * df["consecutive_working_days"]
    - 0.2 * df["tasks_completed"]
    + np.random.normal(0, 4, N)
)

# Convert score into three classes
low_cutoff = risk_score.quantile(0.33)
high_cutoff = risk_score.quantile(0.67)

df["burnout_risk"] = np.select(
    [risk_score <= low_cutoff, risk_score <= high_cutoff],
    [0, 1],
    default=2
)

# Save CSV in the project's data folder
output_path = Path(__file__).resolve().parent / "developer_wellbeing.csv"
df.to_csv(output_path, index=False)

print("DATASET CREATED SUCCESSFULLY!")
print("Shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nRisk class counts:")
print(df["burnout_risk"].value_counts().sort_index())
print("\nSaved to:", output_path)
