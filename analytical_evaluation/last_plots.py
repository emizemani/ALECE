import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/emizemani/Desktop/ALECE/analytical_evaluation/data.csv")

# For demonstration, assume avg query execution time is 1/10th of the E2E time.
df["avg_query_time_ms"] = df["e2e_time_ms"] / 10


# We'll melt the DataFrame so that we have two metrics per experiment.
metrics_df = df[["experiment", "avg_query_time_ms", "e2e_time_ms"]].melt(
    id_vars="experiment",
    var_name="Metric",
    value_name="Time_ms"
)

order_experiments = df["experiment"].tolist()

plt.figure(figsize=(12, 6))
bar_ax = sns.barplot(
    x="experiment", y="Time_ms", hue="Metric",
    data=metrics_df, order=order_experiments, palette="Set2"
)
plt.title("Average Query Execution Time vs. Total E2E Time")
plt.xlabel("Experiment")
plt.ylabel("Time (ms)")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Metric", loc="best")
plt.tight_layout()
plt.savefig("plot_bar_query_e2e_time.png", dpi=300)
plt.close()

# For histograms:
df["hist_mismatch_pct"] = abs(df["hist_dim_target"] - df["hist_dim_source"]) / df["hist_dim_source"] * 100
# For query dimensions:
df["query_mismatch_pct"] = abs(df["query_dim_target"] - df["query_dim_source"]) / df["query_dim_source"] * 100
# Combine (average mismatch percentage)
df["dim_mismatch_pct"] = (df["hist_mismatch_pct"] + df["query_mismatch_pct"]) / 2

# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
# Use regplot to add a trend line
scatter_ax = sns.regplot(
    x="dim_mismatch_pct", y="q_error_50", data=df,
    scatter_kws={"s": 100, "alpha": 0.8}, line_kws={"color": "red"}
)
plt.title("Dimensionality Mismatch vs. Median Q-error (50th Percentile)")
plt.xlabel("Dimensionality Mismatch (%)")
plt.ylabel("Median Q-error (q_error_50)")
# Annotate each point with experiment name for clarity
for i, row in df.iterrows():
    plt.text(row["dim_mismatch_pct"] + 0.5, row["q_error_50"] + 0.5, 
             row["experiment"], fontsize=9, color="black")
plt.tight_layout()
plt.savefig("plot_scatter_dim_vs_qerror.png", dpi=300)
plt.close()
