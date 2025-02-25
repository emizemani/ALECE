import psycopg2

def generate_table_info_txt(database, output_path):
    conn = psycopg2.connect(dbname=database, user='user', password='your_password', host='localhost', port=4321)
    cursor = conn.cursor()

    # Query tables and row counts
    cursor.execute("""
        SELECT table_name, 
               row_number() OVER () - 1 AS table_id
        FROM information_schema.tables 
        WHERE table_schema = 'public';
    """)
    tables = cursor.fetchall()

    # Fetch row counts for each table
    row_counts = {}
    for table_name, table_id in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]
        row_counts[table_name] = row_count

    # Query attribute information for each table
    attr_info = {}
    for table_name, table_id in tables:
        cursor.execute(f"""
            SELECT column_name, ordinal_position - 1 AS attr_no 
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = '{table_name}';
        """)
        columns = cursor.fetchall()

        # Initialize lists to hold types and ranges
        types = []
        ranges = []

        for col, _ in columns:
            # Get the data type
            cursor.execute(f"""
                SELECT data_type 
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = '{table_name}' AND column_name = '{col}';
            """)
            data_type = cursor.fetchone()[0]
            types.append((col, data_type))

            # Get the min and max values for the column, filtering out non-numeric values
            cursor.execute(f"""
                SELECT 
                    MIN(NULLIF("{col}", ''))::float, 
                    MAX(NULLIF("{col}", ''))::float 
                FROM {table_name}
                WHERE "{col}" ~ '^[0-9]+(\.[0-9]+)?$';  -- Regular expression to match numeric values
            """)
            min_val, max_val = cursor.fetchone()
            ranges.append((col, min_val, max_val))

        attr_info[table_name] = {
            'columns': columns,
            'types': types,
            'ranges': ranges
        }

    # Write to table_info.txt
    with open(output_path, 'w') as f:
        for table_name, table_id in tables:
            row_count = row_counts[table_name]
            f.write(f"{table_name},{table_id},{row_count}\n")

            # Write columns
            columns = attr_info[table_name]['columns']
            columns_info = "|".join([f"{col},{pos}" for col, pos in columns])
            f.write(f"{table_name} attr nos: {columns_info}\n")

            # Write types
            types = attr_info[table_name]['types']
            types_info = ",".join([str(t[1]) for t in types])
            f.write(f"{table_name} attr types: {types_info}\n")

            # Write ranges
            ranges = attr_info[table_name]['ranges']
            ranges_info = ",".join([f"{r[1]},{r[2]}" for r in ranges])
            f.write(f"{table_name} attr ranges: {ranges_info}\n")

    cursor.close()
    conn.close()

# Example usage:
generate_table_info_txt("imdbload", "/home/user/Desktop/ALECE/data/IMDB/table_info.txt")
generate_table_info_txt("tpcds", "/home/user/Desktop/ALECE/data/TPCDS/table_info.txt")
