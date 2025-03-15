import psycopg2
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# **Hardcoded file paths**
WORKLOAD_FILE = "/home/emi/ALECE/data/tpch/workload/static/workload.sql"
OUTPUT_FILE = "/home/emi/ALECE/data/tpch/workload/static/sub_queries.sql"
DB_NAME = "tpch"

def read_queries_from_workload():
    """
    Reads train and test queries from the workload.sql file.
    Returns two lists: train_queries and test_queries.
    """
    train_queries = []
    test_queries = []

    with open(WORKLOAD_FILE, "r") as f:
        for line in f:
            match = re.match(r"^(train_query|test_query):\s*(select .*?)\s*;\|\|1\|\|1\|\|\d+\|\|", line, re.IGNORECASE)
            if match:
                query_type, query = match.groups()
                if query_type == "train_query":
                    train_queries.append(query)
                elif query_type == "test_query":
                    test_queries.append(query)

    logging.info(f"Extracted {len(train_queries)} train queries and {len(test_queries)} test queries.")
    return train_queries, test_queries


def generate_and_save_sub_queries(queries, query_type):
    """
    Processes train or test queries:
    - Modifies COUNT(*) queries to SELECT * for accurate cardinality.
    - Runs EXPLAIN (FORMAT JSON) to generate sub-queries.
    - Extracts sub-queries from PostgreSQL.
    - Runs EXPLAIN (FORMAT JSON) on sub-queries to get cardinality.
    - Saves results in the required format.
    """
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(dbname=DB_NAME, user="emi", host="localhost", port=4321)
        print(f"✅ Successfully connected to database: {DB_NAME}")
        cur = conn.cursor()

        # Enable PostgreSQL sub-query generation
        cur.execute("SET print_sub_queries = true;")
        conn.commit()

        with open(OUTPUT_FILE, "a") as f:
            for query in queries:
                logging.info(f"Processing {query_type} query: {query.strip()}")

                # Modify query: replace COUNT(*) with SELECT * to get correct cardinality
                modified_query = re.sub(r"SELECT\s+COUNT\(\*\)", "SELECT *", query, flags=re.IGNORECASE)

                try:
                    # Run EXPLAIN (FORMAT JSON) to extract execution plan
                    print(f"🔵 Running EXPLAIN on modified query: {modified_query.strip()}")
                    cur.execute("EXPLAIN (FORMAT JSON) " + modified_query)
                    explain_result = cur.fetchone()

                    # Extract sub-queries from the execution plan
                    sub_queries = []
                    if explain_result:
                        query_plan = explain_result[0][0]  # Extract the first JSON object
                        if 'Plan' in query_plan:
                            sub_queries.append(modified_query)  # Use the modified query as the only "sub-query" for now

                    if not sub_queries:
                        print(f"⚠️ No sub-queries found for query: {modified_query.strip()}")
                        continue  # Skip if no sub-queries exist

                    # ✅ Run EXPLAIN on extracted sub-queries
                    for sub_query in sub_queries:
                        try:
                            print(f"🔵 Running EXPLAIN on sub-query: {sub_query.strip()}")
                            cur.execute("EXPLAIN (FORMAT JSON) " + sub_query)
                            sub_explain_result = cur.fetchone()

                            # Extract estimated cardinality
                            estimated_cardinality = sub_explain_result[0][0]['Plan']['Plan Rows']

                            # Format and save to file
                            f.write(f"{query_type}_sub_query: {sub_query.strip()} ||1||1||{estimated_cardinality}\n")
                            f.flush()
                            os.fsync(f.fileno())  # Ensure immediate disk write

                            logging.info(f"✅ Saved {query_type} sub-query with cardinality {estimated_cardinality}")

                        except Exception as sub_query_error:
                            print(f"❌ Error processing sub-query: {sub_query.strip()}")
                            print(f"🔴 PostgreSQL Error: {sub_query_error}")
                            logging.error(f"Error processing sub-query: {sub_query_error}")

                except Exception as query_error:
                    print(f"❌ Error processing query: {modified_query.strip()}")
                    print(f"🔴 PostgreSQL Error: {query_error}")
                    conn.rollback()  # Avoid transaction blocking
                    logging.error(f"Error processing {query_type} query: {query_error}")

        cur.close()
        conn.close()
        logging.info(f"Finished processing {query_type} queries.")

    except Exception as e:
        logging.error(f"Database connection failed: {e}")


if __name__ == '__main__':
    # **Step 1: Read Queries from workload.sql**
    print("🚀 Script has started!", flush=True)

    train_queries, test_queries = read_queries_from_workload()

    # **Step 2: Clear output file before writing new results**
    open(OUTPUT_FILE, "w").close()

    # **Step 3: Generate and Save Sub-Queries for Train Queries**
    generate_and_save_sub_queries(train_queries, "train")

    # **Step 4: Generate and Save Sub-Queries for Test Queries**
    generate_and_save_sub_queries(test_queries, "test")

    logging.info("Completed processing all queries.")
