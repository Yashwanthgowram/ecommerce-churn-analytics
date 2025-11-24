import pandas as pd
import os

print("Starting data merge...")

# Define the folder path. '.' means 'the current folder'
# This assumes your 9 CSV files are in the SAME folder as this script.
data_path = '.'

# Load all 9 CSV files
try:
    customers = pd.read_csv(os.path.join(data_path, 'olist_customers_dataset.csv'))
    order_items = pd.read_csv(os.path.join(data_path, 'olist_order_items_dataset.csv'))
    order_payments = pd.read_csv(os.path.join(data_path, 'olist_order_payments_dataset.csv'))
    order_reviews = pd.read_csv(os.path.join(data_path, 'olist_order_reviews_dataset.csv'))
    orders = pd.read_csv(os.path.join(data_path, 'olist_orders_dataset.csv'))
    products = pd.read_csv(os.path.join(data_path, 'olist_products_dataset.csv'))
    sellers = pd.read_csv(os.path.join(data_path, 'olist_sellers_dataset.csv'))
    category_translation = pd.read_csv(os.path.join(data_path, 'product_category_name_translation.csv'))
    
    print("All 9 files loaded into memory.")

except FileNotFoundError as e:
    print(f"Error: {e}. Make sure all 9 CSV files are in the same folder as this script.")
    print("Please download the data from Kaggle, unzip it, and place the CSVs here.")
    exit() # Stop the script if files are missing

# --- Start merging ---
# We'll merge everything into the 'orders' table

# 1. Merge orders with order_items (one order can have multiple items)
merged_df = pd.merge(orders, order_items, on='order_id', how='left')

# 2. Merge with order_payments (one order can have multiple payment methods)
merged_df = pd.merge(merged_df, order_payments, on='order_id', how='left')

# 3. Merge with order_reviews
merged_df = pd.merge(merged_df, order_reviews, on='order_id', how='left')

# 4. Merge with products
merged_df = pd.merge(merged_df, products, on='product_id', how='left')

# 5. Merge with customers (to get customer_unique_id and location)
merged_df = pd.merge(merged_df, customers, on='customer_id', how='left')

# 6. Merge with sellers
merged_df = pd.merge(merged_df, sellers, on='seller_id', how='left')

# 7. Merge with category_translation (to get English names)
merged_df = pd.merge(merged_df, category_translation, on='product_category_name', how='left')

print("All files successfully merged.")

# --- Save the final master dataset ---
# This creates the 'olist_master_dataset.csv' file you need for the next steps
merged_df.to_csv('olist_master_dataset.csv', index=False)

print(f"SUCCESS: Master dataset saved as 'olist_master_dataset.csv'")
print(f"It has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns.")