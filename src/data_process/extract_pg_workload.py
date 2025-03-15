def transform_workload(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Skip lines that are not queries
            if not line.startswith(('train_query:', 'test_query:')):
                continue

            # Split the line into query and cardinality parts
            query_part, cardinality_part = line.split('||', 1)
            query = query_part.split(':', 1)[1].strip()
            cardinalities = cardinality_part.split(',')

            # Extract the true cardinality (the first value in the cardinalities)
            true_cardinality = cardinalities[0].strip()

            # Construct the new query format
            new_query = f"{query_part.split(':')[0]}: select count(*) {query.split('from', 1)[1]};||1||1||{true_cardinality}||\n"
            
            # Write the transformed query to the output file
            outfile.write(new_query)

# Specify the input and output file paths
input_file = 'data/tpch/workload/static/mysql_workload.sql'
output_file = 'data/tpch/workload/static/workload.sql'

# Transform the workload
transform_workload(input_file, output_file)