import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/emizemani/Desktop/ALECE/analytical_evaluation/data.csv")

sns.set_theme(style="whitegrid")

# We want the experiments grouped by target dataset, in the exact order:
#  1) JL_scratch, TPCH_to_JL, ST_to_JL
#  2) ST_scratch, TPCH_to_ST, JL_to_ST
#  3) TPCH_scratch, JL_to_TPCH, ST_to_TPCH
custom_order = [
    "JL_scratch", "TPCH_to_JL", "ST_to_JL",
    "ST_scratch", "TPCH_to_ST", "JL_to_ST",
    "TPCH_scratch", "JL_to_TPCH", "ST_to_TPCH"
]

# Create a quick map from scratch experiments to the dataset code
# so we can identify baselines.
scratch_map = {
    "JL_scratch":  "JL",
    "ST_scratch":  "ST",
    "TPCH_scratch": "TPCH"
}

# Identify each row's target dataset by parsing the experiment name.
# We'll need this for calculating % difference from the correct baseline.
def get_target_dataset(exp_name):
    if exp_name in scratch_map:
        return scratch_map[exp_name]
    # Transfer notation: e.g. 'JL_to_ST' => target = 'ST'
    parts = exp_name.split("_to_")
    if len(parts) == 2:
        return parts[1]  # the part after '_to_'
    return "Unknown"

df["target_dataset"] = df["experiment"].apply(get_target_dataset)

# Build a dict of baseline training times: for each dataset code, find the scratch row
baseline_times = {}
for scratch_exp, ds_code in scratch_map.items():
    row = df[df["experiment"] == scratch_exp]
    if not row.empty:
        baseline_times[ds_code] = float(row["training_time"].values[0])

# Calculate percentage difference from the relevant scratch experiment
def compute_time_diff(row):
    ds = row["target_dataset"]
    if ds in baseline_times:
        baseline = baseline_times[ds]
        current = row["training_time"]
        # If the experiment itself is scratch, no difference (0.0)
        if row["experiment"] in scratch_map:
            return 0.0
        return ((current - baseline) / baseline) * 100
    return 0.0

df["train_time_diff_pct"] = df.apply(compute_time_diff, axis=1)


plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x="experiment",
    y="training_time",
    data=df,
    order=custom_order,
    palette="Set2"
)
plt.title("Training Time by Experiment (Grouped by Target Dataset)")
plt.ylabel("Training Time (Hours)")
plt.xlabel("Experiment")
plt.xticks(rotation=45, ha="right")

# Add vertical lines to separate each group of 3 bars:
# Group 1: indices 0,1,2
# Group 2: indices 3,4,5
# Group 3: indices 6,7,8
# We'll place lines at x=2.5 and x=5.5
for line_x in [2.5, 5.5]:
    plt.axvline(x=line_x, color="black", linestyle="--", alpha=0.5)

# Annotate each bar (except scratch) with % difference from scratch
# We'll read the actual positions from the bar container
for i, bar in enumerate(ax.patches):
    exp_name = custom_order[i]  # which experiment is this
    # If it's scratch, skip annotation
    if exp_name in scratch_map:
        continue
    height = bar.get_height()
    diff_pct = df.loc[df["experiment"] == exp_name, "train_time_diff_pct"].values[0]
    ax.text(
        bar.get_x() + bar.get_width()/2, 
        height + 0.1, 
        f"{diff_pct:+.1f}%", 
        ha="center", va="bottom", fontsize=9, color="black"
    )

plt.tight_layout()
plt.savefig("plot1_training_time_grouped.png", dpi=300)
plt.close()

# plot bar 2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# CPU usage subplot
sns.barplot(
    x="experiment",
    y="cpu",
    data=df,
    order=custom_order,
    palette="Set2",
    ax=ax1
)
ax1.set_title("CPU Usage (%)")
ax1.set_xlabel("Experiment")
ax1.set_ylabel("CPU (%)")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")

# Draw vertical lines to separate each group of 3
for line_x in [2.5, 5.5]:
    ax1.axvline(x=line_x, color="black", linestyle="--", alpha=0.5)

# Memory usage subplot
sns.barplot(
    x="experiment",
    y="memory_mb",
    data=df,
    order=custom_order,
    palette="Set2",
    ax=ax2
)
ax2.set_title("Memory Usage (MB)")
ax2.set_xlabel("Experiment")
ax2.set_ylabel("Memory (MB)")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")

for line_x in [2.5, 5.5]:
    ax2.axvline(x=line_x, color="black", linestyle="--", alpha=0.5)

fig.suptitle("Resource Consumption by Experiment", fontsize=14)
fig.tight_layout()
fig.savefig("plot2_resource_consumption.png", dpi=300)
plt.close()

# plot 3
# We'll color by baseline vs. transfer and label each point with the experiment name.
def is_transfer(exp):
    return "Transfer" if exp not in scratch_map else "Baseline"

df["exp_type"] = df["experiment"].apply(is_transfer)

plt.figure(figsize=(8, 6))
scatter_ax = sns.scatterplot(
    x="training_time",
    y="memory_mb",
    hue="exp_type",
    data=df,
    s=100,
    palette=["blue", "orange"]
)

plt.title("Training Time vs. Memory Usage")
plt.xlabel("Training Time (Hours)")
plt.ylabel("Memory Usage (MB)")
plt.legend(title="Experiment Type", loc="best")

# Add text labels for each point so we know exactly which experiment it is
offset_x = 0.2
offset_y = 50
for i, row in df.iterrows():
    x_val = row["training_time"]
    y_val = row["memory_mb"]
    exp_label = row["experiment"]
    plt.text(
        x_val + offset_x, 
        y_val + offset_y, 
        exp_label,
        fontsize=8,
        color="black"
    )

plt.tight_layout()
plt.savefig("plot3_training_vs_memory_labeled.png", dpi=300)
plt.close()
