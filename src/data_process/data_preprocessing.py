def transform_query(line):
    # Skip if not a query line
    if not ('train_query:' in line or 'test_query:' in line):
        return None
        
    # Split into query and results
    parts = line.strip().split('||')
    query_part = parts[0]
    results = parts[1]  # The last part will be empty because line ends with ||
    
    tag, full_query = query_part.split(': ', 1)
    
    # Extract just the COUNT(*) query by removing other aggregations
    count_query = full_query[:full_query.find(', avg')] + full_query[full_query.find(' from'):]
    
    # Get the first number from results (the COUNT(*) result)
    count_result = results.split(',')[0]
    
    # Format in the required way
    return f"{tag}: {count_query}||X||Y||{count_result}||"

def process_workload_file(input_path, train_output_path, test_output_path):
    train_queries = []
    test_queries = []
    
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('load data') and not line.startswith('--'):
                transformed = transform_query(line)
                if transformed:
                    if transformed.startswith('train_query:'):
                        train_queries.append(transformed)
                    elif transformed.startswith('test_query:'):
                        test_queries.append(transformed)
    
    # Write train queries
    with open(train_output_path, 'w') as f:
        f.write('\n'.join(train_queries))
        
    # Write test queries
    with open(test_output_path, 'w') as f:
        f.write('\n'.join(test_queries))


input_job_light = "/Users/emizemani/Desktop/ALECE/data/job_light/workload/static/mysql_workload.sql"
input_tpch = "/Users/emizemani/Desktop/ALECE/data/tpch/workload/static/mysql_workload.sql"
train_output_job_light = "/Users/emizemani/Desktop/ALECE/data/job_light/workload/static/train_queries.sql"
test_output_job_light = "/Users/emizemani/Desktop/ALECE/data/job_light/workload/static/test_queries.sql"
train_output_tpch = "/Users/emizemani/Desktop/ALECE/data/tpch/workload/static/train_queries.sql"
test_output_tpch = "/Users/emizemani/Desktop/ALECE/data/tpch/workload/static/test_queries.sql"

process_workload_file(input_job_light, train_output_job_light, test_output_job_light)
process_workload_file(input_tpch, train_output_tpch, test_output_tpch)