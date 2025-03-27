import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/emizemani/Desktop/ALECE/analytical_evaluation/data.csv")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# training times
# Define mapping for scratch experiments.
scratch_map = {
    "JL_scratch":  "JL",
    "ST_scratch":  "ST",
    "TPCH_scratch": "TPCH"
}

def get_target_dataset(exp_name):
    if exp_name in scratch_map:
        return scratch_map[exp_name]
    parts = exp_name.split("_to_")
    if len(parts) == 2:
        return parts[1]  # target dataset code
    return "Unknown"

df["target_dataset"] = df["experiment"].apply(get_target_dataset)

# Create a dict of baseline training times for each dataset.
baseline_times = {}
for scratch_exp, ds_code in scratch_map.items():
    row = df[df["experiment"] == scratch_exp]
    if not row.empty:
        baseline_times[ds_code] = float(row["training_time"].values[0])


def compute_time_reduction(row):
    ds = row["target_dataset"]
    if ds in baseline_times:
        baseline = baseline_times[ds]
        current = row["training_time"]
        # For scratch experiments, reduction is 0.
        if row["experiment"] in scratch_map:
            return 0.0
        return ((baseline - current) / baseline) * 100
    return 0.0

df["train_time_reduction_pct"] = df.apply(compute_time_reduction, axis=1)

def label_model_type(exp_name):
    return "Baseline" if exp_name in scratch_map else "Transfer"

df["model_type"] = df["experiment"].apply(label_model_type)

# Q Error line
qerr_cols = ["q_error_50", "q_error_90", "q_error_95", "q_error_99"]
df_qerr = df.melt(id_vars="experiment", value_vars=qerr_cols,
                  var_name="Percentile", value_name="Q_error")

plt.figure(figsize=(12, 6))
line_ax = sns.lineplot(
    x="experiment", y="Q_error", hue="Percentile",
    style="Percentile", markers=True, dashes=False,
    data=df_qerr, palette="Set1", linewidth=2
)
plt.title("Q-error Percentiles Across Experiments")
plt.xlabel("Experiment")
plt.ylabel("Q-error")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("plot_line_qerror.png", dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# Scatter Plot
# We'll use q_error_50 as the median Q-error.
plt.figure(figsize=(10, 6))
scatter_ax = sns.scatterplot(
    x="train_time_reduction_pct", y="q_error_50",
    hue="model_type", style="model_type",
    data=df, s=120, palette=["blue", "orange"]
)
plt.title("Training Time Reduction vs. Median (50th) Q-error")
plt.xlabel("Training Time Reduction (%)")
plt.ylabel("Median Q-error (50th Percentile)")
# Annotate each point with experiment name
for i, row in df.iterrows():
    scatter_ax.text(row["train_time_reduction_pct"] + 0.5,
                    row["q_error_50"] + 0.5,
                    row["experiment"],
                    horizontalalignment='left', size='small', color='black', weight='semibold')
plt.tight_layout()
plt.savefig("plot_scatter_train_vs_qerror.png", dpi=300)
plt.close()

# Heat Map
heatmap_df = df.set_index("experiment")[qerr_cols]

plt.figure(figsize=(10, 6))
heat_ax = sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Q-error'})
plt.title("Heatmap of Q-error Percentiles Across Experiments")
plt.xlabel("Q-error Percentile")
plt.ylabel("Experiment")
plt.tight_layout()
plt.savefig("plot_heatmap_qerror.png", dpi=300)
plt.close()
