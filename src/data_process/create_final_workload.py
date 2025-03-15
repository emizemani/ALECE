def merge_workload_files():
    # Paths
    base_dir = "/Users/emizemani/Desktop/ALECE/data/tpch/workload/static"
    workload_path = f"{base_dir}/workload.sql"
    sub_queries_path = f"{base_dir}/sub_queries.sql"
    output_path = f"{base_dir}/workload_with_sub_queries.sql"

    # Read workload queries
    with open(workload_path, 'r') as f:
        workload_lines = f.readlines()

    # Read sub-queries
    with open(sub_queries_path, 'r') as f:
        sub_queries_lines = f.readlines()

    # Find where test queries start
    test_start_idx = 0
    for i, line in enumerate(workload_lines):
        if line.startswith('test_query:'):
            test_start_idx = i
            break

    # Create new workload file
    new_workload = []
    new_workload.extend(workload_lines[:test_start_idx])    # All train queries
    new_workload.extend(sub_queries_lines)                  # Add all sub-queries
    new_workload.extend(workload_lines[test_start_idx:])    # All test queries

    # Backup original workload file
    import shutil
    shutil.copy2(workload_path, f"{workload_path}.backup")

    # Write new workload file
    with open(output_path, 'w') as f:
        f.writelines(new_workload)

    print(f"Created new workload file at: {output_path}")
    print(f"Original workload backed up at: {workload_path}.backup")

if __name__ == "__main__":
    merge_workload_files()