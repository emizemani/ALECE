import numpy as np
import json

# Load meta_infos.npy to retrieve dimensions for STATS dataset
stats_meta_infos = np.load('/Users/emizemani/Desktop/ALECE/data/stats/workload/static/features/feature_meta_infos.npy')

# The meta_infos contain: [histogram_feature_dim, num_attrs, n_possible_joins]
# We can convert it to dict format to store in dim_info.json
stats_dim_info = {
    "histogram_feature_dim": int(stats_meta_infos[0]),
    "query_part_feature_dim": int(stats_meta_infos[1]),
    "join_pattern_dim": int(stats_meta_infos[2])
}

# Save dim_info.json in the baseline ckpt folder for STATS
with open("/Users/emizemani/Desktop/ALECE_trained_from_scratch/exp/STATS/dims_info.json", "w") as f:
    json.dump(stats_dim_info, f, indent=4)

print("✅ dims_info.json for STATS saved successfully!")

# Repeat for JOB-light dataset
job_light_meta_infos = np.load('/home/emi/ALECE/data/job_light/workload/static/features/feature_meta_infos.npy')

job_light_dim_info = {
    "histogram_feature_dim": int(job_light_meta_infos[0]),
    "query_part_feature_dim": int(job_light_meta_infos[1]),
    "join_pattern_dim": int(job_light_meta_infos[2])
}

with open("../exp/job_light/dims_info.json", "w") as f:
    json.dump(job_light_dim_info, f, indent=4)

print("✅ dims_info.json for JOB-light saved successfully!")

# Repeat for TPC-H dataset
tpch_meta_infos = np.load('/home/emi/ALECE/data/tpch/workload/static/features/feature_meta_infos.npy')

tpch_dim_info = {
    "histogram_feature_dim": int(tpch_meta_infos[0]),
    "query_part_feature_dim": int(tpch_meta_infos[1]),
    "join_pattern_dim": int(tpch_meta_infos[2])
}

with open("../exp/tpch/dims_info.json", "w") as f:
    json.dump(tpch_dim_info, f, indent=4)

print("✅ dims_info.json for TPC-H saved successfully!")
