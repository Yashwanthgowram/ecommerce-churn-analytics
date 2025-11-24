import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns

# Set matplotlib backend to Agg to avoid GUI errors on servers/some setups
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Starting Step 2: Cleaning and Feature Engineering...")

# --- 1. LOAD DATA ---
try:
    df = pd.read_csv('olist_master_dataset.csv')
    print(f"Loaded 'olist_master_dataset.csv'. It has {df.shape[0]} rows.")
except FileNotFoundError:
    print("Error: 'olist_master_dataset.csv' not found.")
    print("Please run the 'merge_data.py' script first.")
    exit()

# --- 2. DATA CLEANING ---

# Convert all date columns from 'object' (text) to 'datetime'
print("Cleaning date columns...")
date_columns = [
    'order_purchase_timestamp', 'order_approved_at', 
    'order_delivered_carrier_date', 'order_delivered_customer_date', 
    'order_estimated_delivery_date', 'review_creation_date', 
    'review_answer_timestamp', 'shipping_limit_date'
]
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Fill missing (NaN) product categories with the text 'unknown'
print("Cleaning missing category names...")
df['product_category_name_english'].fillna('unknown', inplace=True)
# We also fill the original portuguese column just in case
df['product_category_name'].fillna('unknown', inplace=True)

# Drop rows where critical IDs are missing (this is rare but good practice)
df.dropna(subset=['order_id', 'customer_unique_id', 'product_id'], inplace=True)


# --- 3. FEATURE ENGINEERING (GOAL 1: RFM DATA) ---
print("Building RFM dataset for churn model...")

# Find the most recent purchase date in the entire dataset
max_purchase_date = df['order_purchase_timestamp'].max()

# Group the data by each unique customer
rfm_data = df.groupby('customer_unique_id').agg(
    # Recency: How many days ago was their last purchase?
    Recency=('order_purchase_timestamp', lambda x: (max_purchase_date - x.max()).days),
    # Frequency: How many total orders have they placed?
    Frequency=('order_id', 'nunique'),
    # Monetary: How much total money have they spent?
    Monetary=('payment_value', 'sum')
).reset_index()

# Save this new RFM dataset to its own file
rfm_data.to_csv('customer_rfm_data.csv', index=False)
print(f"SUCCESS: Saved 'customer_rfm_data.csv' with {rfm_data.shape[0]} customers.")


# --- 4. FEATURE ENGINEERING (GOAL 2: FORECAST DATA) ---
print("Building daily sales dataset for forecast model...")

# First, we need to get just the date part of the timestamp (not the time)
df['order_purchase_date'] = df['order_purchase_timestamp'].dt.date

# Now, group by that date and sum the sales for each day
daily_sales = df.groupby('order_purchase_date')['payment_value'].sum().reset_index()

# Rename the columns to 'ds' (date) and 'y' (value) as required by the Prophet model
daily_sales.rename(columns={'order_purchase_date': 'ds', 'payment_value': 'y'}, inplace=True)

# Save this new sales dataset to its own file
daily_sales.to_csv('daily_sales_forecast_data.csv', index=False)
print(f"SUCCESS: Saved 'daily_sales_forecast_data.csv' with {daily_sales.shape[0]} days of sales.")

print("\nStep 2 Complete!")