import sqlglot
import csv

datasets = {
    "stats": "/Users/emizemani/Desktop/ALECE/data/stats/workload/static/workload.sql",
    "job_light": "/Users/emizemani/Desktop/ALECE/data/job_light/workload/static/workload.sql",
    "tpch": "/Users/emizemani/Desktop/ALECE/data/tpch/workload/static/workload.sql"
}

def is_valid_query_line(line):
    return not (line.strip().startswith("COPY") or line.strip().startswith("-"))

def extract_query(line):
    if ":" in line:
        query = line.split(":", 1)[1].strip()
        if "||" in query:
            query = query.split("||", 1)[0].strip()
        return query
    return None


def structured_join_count(sql, debug=False):
    try:
        parsed = sqlglot.parse(sql)
        join_count = 0
        for expression in parsed:
            joins = expression.find_all(sqlglot.expressions.Join)
            join_count += len(list(joins))
        return join_count
    except Exception as e:
        if debug:
            print(f"Failed to parse:\n{sql}\nError: {e}\n")
        return 0


def analyze_queries(filepath):
    queries = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if is_valid_query_line(line):
                query = extract_query(line)
                if query:
                    queries.append(query)

    join_counts = [structured_join_count(q) for q in queries if q]
    if not join_counts:
        return {
            "nr_of_queries": 0,
            "avg_joins_per_query": 0,
            "max_joins_per_query": 0
        }

    avg_joins = sum(join_counts) / len(join_counts)
    max_joins = max(join_counts)

    return {
        "nr_of_queries": len(queries),
        "avg_joins_per_query": round(avg_joins, 2),
        "max_joins_per_query": max_joins
    }

results = []

for dataset_name, path in datasets.items():
    metrics = analyze_queries(path)
    metrics["dataset"] = dataset_name
    results.append(metrics)

# === Save results to CSV ===
output_file = "query_workload_summary.csv"
fieldnames = ["dataset", "nr_of_queries", "avg_joins_per_query", "max_joins_per_query"]

with open(output_file, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Saved summary to {output_file}")
