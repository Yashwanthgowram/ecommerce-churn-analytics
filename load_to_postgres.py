import pandas as pd
from sqlalchemy import create_engine
import time

print("Starting data load to Postgres...")

# This 'connection string' tells Python how to connect to your new database
# It uses the username, password, and DB name you set in your docker command
db_url = 'postgresql://myuser:mysecretpassword@localhost:5432/olist_db'

try:
    engine = create_engine('postgresql://postgres:Y%40shwanth0317@localhost:5432/ecommerce_db')
    print("Database engine created successfully.")
except Exception as e:
    print(f"Error creating engine: {e}")
    print("Please make sure your Postgres container is running.")
    exit()

# List of your 3 CSV files and the table names you want to create
files_to_load = {
    'olist_master_dataset.csv': 'olist_master_dataset',
    'customer_rfm_data.csv': 'customer_rfm_data',
    'sales_forecast_output.csv': 'sales_forecast_output'
}

print(f"Found {len(files_to_load)} files to load...")
start_time = time.time()

for csv_file, table_name in files_to_load.items():
    print(f"Loading '{csv_file}' into table '{table_name}'...")
    try:
        # Read the CSV file from your project folder
        df = pd.read_csv(csv_file)
        
        # This command loads the DataFrame into a new SQL table
        # 'if_exists='replace'' will drop the table if it already exists (good for re-running)
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        
        print(f"Successfully loaded '{csv_file}'.")
    
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found. Please make sure it's in the same folder.")
    except Exception as e:
        print(f"Error loading '{csv_file}': {e}")

end_time = time.time()
print(f"\nAll data loaded successfully in {end_time - start_time:.2f} seconds.")
print("You can now connect Metabase or pgAdmin to your database.")