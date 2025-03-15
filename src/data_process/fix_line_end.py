def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Flag to start processing lines after the static marker
    process_lines = False
    updated_lines = []

    for line in lines:
        # Check for the static marker
        if '---------------static---------------' in line:
            process_lines = True
            updated_lines.append(line)
            continue

        # Process lines after the static marker
        if process_lines:
            # Strip newline characters for processing
            stripped_line = line.rstrip('\n')
            # Check if the line ends with '||'
            if not stripped_line.endswith('||'):
                stripped_line += '||'
            updated_lines.append(stripped_line + '\n')
        else:
            updated_lines.append(line)

    # Write the updated lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(updated_lines)

# Specify the path to your file
file_path = '/Users/emizemani/Desktop/ALECE/data/tpch/workload/static/workload.sql'
process_file(file_path)