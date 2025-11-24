# Data Dictionary - E-Commerce Churn Analytics

**Dataset Source**: Brazilian E-Commerce Public Dataset by Olist (Kaggle)
**Period**: September 2016 - October 2018
**Records**: 100,000+ orders
**Last Updated**: 2025-11-24

---

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Source Datasets (9 CSV Files)](#source-datasets)
3. [Derived Datasets](#derived-datasets)
4. [Database Tables](#database-tables)
5. [Data Relationships](#data-relationships)
6. [Data Quality Rules](#data-quality-rules)

---

## 1. Dataset Overview

The Olist dataset consists of information on 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil. The data is divided across 9 CSV files with relationships that can be joined to create a comprehensive view of e-commerce transactions.

### Key Entities
- **Orders**: 99,441 unique orders
- **Customers**: 99,441 unique customers
- **Products**: 32,951 unique products
- **Sellers**: 3,095 unique sellers
- **Categories**: 73 product categories

---

## 2. Source Datasets (9 CSV Files)

### 2.1 olist_orders_dataset.csv

**Description**: Core order information including order lifecycle timestamps

| Column Name | Data Type | Description | Example | Nullable |
|-------------|-----------|-------------|---------|----------|
| order_id | VARCHAR(50) | Unique identifier for order | 8d50f204bdacbd4e7c36f9a9b0e7e6d3 | No |
| customer_id | VARCHAR(50) | Unique identifier for customer | 5e274e7a4b5e8fdf7e3e06e9089b4f3c | No |
| order_status | VARCHAR(20) | Current status of order | delivered, shipped, canceled | No |
| order_purchase_timestamp | TIMESTAMP | When order was placed | 2017-10-02 10:56:33 | No |
| order_approved_at | TIMESTAMP | When payment was approved | 2017-10-02 11:07:15 | Yes |
| order_delivered_carrier_date | TIMESTAMP | When order handed to carrier | 2017-10-04 19:55:00 | Yes |
| order_delivered_customer_date | TIMESTAMP | When customer received order | 2017-10-10 21:25:13 | Yes |
| order_estimated_delivery_date | DATE | Estimated delivery date | 2017-10-18 | Yes |

**Order Status Values**:
- `delivered` (96.5%)
- `shipped` (1.1%)
- `canceled` (0.6%)
- `unavailable` (0.6%)
- `invoiced` (0.3%)
- `processing` (0.3%)
- Others (0.6%)

---

### 2.2 olist_customers_dataset.csv

**Description**: Customer master data including location information

| Column Name | Data Type | Description | Example | Nullable |
|-------------|-----------|-------------|---------|----------|
| customer_id | VARCHAR(50) | Unique identifier per order | 06b8999e2fba1a1fbc88172c00ba8bc7 | No |
| customer_unique_id | VARCHAR(50) | Unique identifier per customer | 861eff4711a542e4b93843c6dd7febb0 | No |
| customer_zip_code_prefix | INTEGER | First 5 digits of zip code | 14409 | No |
| customer_city | VARCHAR(50) | Customer city | sao paulo | No |
| customer_state | VARCHAR(2) | Customer state (2-letter code) | SP | No |

**Note**: `customer_id` is unique per order, while `customer_unique_id` tracks repeat customers across orders.

**Top States**:
- SP (São Paulo): 41.7%
- RJ (Rio de Janeiro): 12.9%
- MG (Minas Gerais): 11.7%
- RS (Rio Grande do Sul): 5.5%
- PR (Paraná): 5.0%

---

### 2.3 olist_order_items_dataset.csv

**Description**: Line-item details for each order (one row per product in an order)

| Column Name | Data Type | Description | Example | Nullable |
|-------------|-----------|-------------|---------|----------|
| order_id | VARCHAR(50) | Reference to order | f00e7a7b15ae7f1ac4b9ca7be8e9a6a3 | No |
| order_item_id | INTEGER | Sequential number of item in order | 1, 2, 3 | No |
| product_id | VARCHAR(50) | Unique identifier for product | 4244733e06e7ecb4970a6e2683c13e61 | No |
| seller_id | VARCHAR(50) | Unique identifier for seller | 48436dade18ac8b2bce089ec2a041202 | No |
| shipping_limit_date | TIMESTAMP | Seller shipping deadline | 2017-04-09 19:13:54 | No |
| price | DECIMAL(10,2) | Item price in BRL | 58.90 | No |
| freight_value | DECIMAL(10,2) | Shipping cost in BRL | 13.29 | No |

**Statistics**:
- Average items per order: 1.2
- Price range: 0.85 - 6,735.00 BRL
- Average freight: 19.99 BRL

---

### 2.4 olist_order_payments_dataset.csv

**Description**: Payment information for orders (one order can have multiple payment methods)

| Column Name | Data Type | Description | Example | Nullable |
|-------------|-----------|-------------|---------|----------|
| order_id | VARCHAR(50) | Reference to order | 36c7e61e28d321fcbbe9c3b9e3a8b78b | No |
| payment_sequential | INTEGER | Sequential number of payment | 1, 2, 3 | No |
| payment_type | VARCHAR(20) | Payment method | credit_card, boleto, voucher | No |
| payment_installments | INTEGER | Number of installments | 1 to 24 | No |
| payment_value | DECIMAL(10,2) | Total payment value | 141.70 | No |

**Payment Types Distribution**:
- `credit_card`: 73.9%
- `boleto`: 19.6% (Brazilian bank slip)
- `voucher`: 5.5%
- `debit_card`: 1.5%

**Installments**:
- 1x (no installment): 52.7%
- 2x-6x: 35.2%
- 7x-12x: 10.8%
- 13x-24x: 1.3%

---

### 2.5 olist_order_reviews_dataset.csv

**Description**: Customer reviews and ratings for orders

| Column Name | Data Type | Description | Example | Nullable |
|-------------|-----------|-------------|---------|----------|
| review_id | VARCHAR(50) | Unique identifier for review | 8485e43a73e9e09fdbe9ab3e3c5f1cd4 | No |
| order_id | VARCHAR(50) | Reference to order | 8d50f204bdacbd4e7c36f9a9b0e7e6d3 | No |
| review_score | INTEGER | Rating from 1 to 5 | 4 | No |
| review_comment_title | TEXT | Review title | "Produto excelente" | Yes |
| review_comment_message | TEXT | Review text | "Chegou rápido e bem embalado" | Yes |
| review_creation_date | TIMESTAMP | When review was created | 2017-10-07 21:56:36 | No |
| review_answer_timestamp | TIMESTAMP | When review was answered | 2017-10-08 10:15:22 | Yes |

**Rating Distribution**:
- 5 stars: 57.8%
- 4 stars: 19.3%
- 1 star: 11.4%
- 3 stars: 8.1%
- 2 stars: 3.3%

**Average Rating**: 4.09 / 5.0

---

### 2.6 olist_products_dataset.csv

**Description**: Product catalog with dimensions and category

| Column Name | Data Type | Description | Example | Nullable |
|-------------|-----------|-------------|---------|----------|
| product_id | VARCHAR(50) | Unique identifier for product | 1e9e8ef04dbcff4541ed26657ea517e5 | No |
| product_category_name | VARCHAR(50) | Category in Portuguese | beleza_saude | Yes |
| product_name_length | INTEGER | Length of product name | 40 | Yes |
| product_description_length | INTEGER | Length of product description | 287 | Yes |
| product_photos_qty | INTEGER | Number of product photos | 1 | Yes |
| product_weight_g | INTEGER | Product weight in grams | 225 | Yes |
| product_length_cm | INTEGER | Product length in cm | 16 | Yes |
| product_height_cm | INTEGER | Product height in cm | 10 | Yes |
| product_width_cm | INTEGER | Product width in cm | 14 | Yes |

**Top Categories**:
1. bed_bath_table (3,029 products)
2. health_beauty (2,444)
3. sports_leisure (2,867)
4. furniture_decor (2,657)
5. computers_accessories (1,639)

---

### 2.7 olist_sellers_dataset.csv

**Description**: Seller master data with location

| Column Name | Data Type | Description | Example | Nullable |
|-------------|-----------|-------------|---------|----------|
| seller_id | VARCHAR(50) | Unique identifier for seller | 3442f8959a84dea7ee197c632cb2df15 | No |
| seller_zip_code_prefix | INTEGER | First 5 digits of zip code | 13023 | No |
| seller_city | VARCHAR(50) | Seller city | campinas | No |
| seller_state | VARCHAR(2) | Seller state (2-letter code) | SP | No |

**Seller Distribution**:
- SP (São Paulo): 52.3%
- PR (Paraná): 7.4%
- MG (Minas Gerais): 7.0%
- RJ (Rio de Janeiro): 5.1%

---

### 2.8 olist_geolocation_dataset.csv

**Description**: Geographic coordinates for Brazilian zip codes

| Column Name | Data Type | Description | Example | Nullable |
|-------------|-----------|-------------|---------|----------|
| geolocation_zip_code_prefix | INTEGER | First 5 digits of zip code | 1037 | No |
| geolocation_lat | DECIMAL(10,6) | Latitude | -23.545621 | No |
| geolocation_lng | DECIMAL(10,6) | Longitude | -46.639292 | No |
| geolocation_city | VARCHAR(50) | City name | sao paulo | No |
| geolocation_state | VARCHAR(2) | State code | SP | No |

**Records**: 1,000,163 lat/lng coordinates
**Use Case**: Geographic visualization and distance calculations

---

### 2.9 product_category_name_translation.csv

**Description**: Translation of product categories from Portuguese to English

| Column Name | Data Type | Description | Example | Nullable |
|-------------|-----------|-------------|---------|----------|
| product_category_name | VARCHAR(50) | Category in Portuguese | beleza_saude | No |
| product_category_name_english | VARCHAR(50) | Category in English | health_beauty | No |

**Total Categories**: 71 translated categories

**Examples**:
- `beleza_saude` → health_beauty
- `informatica_acessorios` → computers_accessories
- `moveis_decoracao` → furniture_decor
- `esporte_lazer` → sports_leisure

---

## 3. Derived Datasets

### 3.1 olist_master_dataset.csv

**Description**: Merged dataset combining all 9 source files

**Creation Script**: `merge_data.py`

**Size**: ~500 MB, 120,000+ rows, 50+ columns

**Key Columns** (selection):
- All columns from orders table
- Customer location (city, state)
- Product details (category, dimensions)
- Seller location
- Payment information
- Review scores

**Use Cases**:
- Comprehensive analysis
- Feature engineering
- Model training input

---

### 3.2 customer_rfm_data.csv

**Description**: Customer-level RFM (Recency, Frequency, Monetary) analysis

**Creation Script**: `explore_and_clean.py`

| Column Name | Data Type | Description | Example | Nullable |
|-------------|-----------|-------------|---------|----------|
| customer_unique_id | VARCHAR(50) | Unique customer identifier | 861eff4711a542e4b93843c6dd7febb0 | No |
| Recency | INTEGER | Days since last purchase | 135 | No |
| Frequency | INTEGER | Number of orders | 2 | No |
| Monetary | DECIMAL(10,2) | Total lifetime value | 245.50 | No |
| churned | INTEGER | Churn label (0 or 1) | 1 | No |

**Calculation Logic**:
```python
# Recency: Days since last purchase
max_date = df['order_purchase_timestamp'].max()
last_purchase = df.groupby('customer_unique_id')['order_purchase_timestamp'].max()
recency = (max_date - last_purchase).dt.days

# Frequency: Number of distinct orders
frequency = df.groupby('customer_unique_id')['order_id'].nunique()

# Monetary: Total spending
monetary = df.groupby('customer_unique_id')['payment_value'].sum()

# Churned: 1 if Recency > 180 days, else 0
churned = (recency > 180).astype(int)
```

**Statistics**:
- Average Recency: 147 days
- Average Frequency: 1.1 orders
- Average Monetary: $153.47
- Churn Rate: 32.4%

---

### 3.3 daily_sales_forecast_data.csv

**Description**: Time-series data for sales forecasting

**Creation Script**: `explore_and_clean.py`

| Column Name | Data Type | Description | Example | Nullable |
|-------------|-----------|-------------|---------|----------|
| ds | DATE | Date | 2017-10-02 | No |
| y | DECIMAL(10,2) | Total sales for date | 15234.50 | No |

**Format**: Prophet-compatible (ds/y columns)
**Date Range**: 2016-09-04 to 2018-10-17
**Use Case**: Sales forecasting with Prophet model

---

### 3.4 sales_forecast_output.csv

**Description**: Forecasted sales for next 6 months

**Creation Script**: `train_models.py`

| Column Name | Data Type | Description | Example | Nullable |
|-------------|-----------|-------------|---------|----------|
| ds | DATE | Forecast date | 2018-11-01 | No |
| yhat | DECIMAL(10,2) | Predicted sales | 18500.23 | No |
| yhat_lower | DECIMAL(10,2) | Lower confidence bound (95%) | 14200.15 | No |
| yhat_upper | DECIMAL(10,2) | Upper confidence bound (95%) | 22800.45 | No |

**Forecast Horizon**: 180 days (6 months)
**Confidence Level**: 95%

---

## 4. Database Tables

### 4.1 PostgreSQL Schema

#### Table: olist_orders
```sql
CREATE TABLE olist_orders (
    order_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    order_status VARCHAR(20) NOT NULL,
    order_purchase_timestamp TIMESTAMP NOT NULL,
    order_approved_at TIMESTAMP,
    order_delivered_carrier_date TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date DATE,
    FOREIGN KEY (customer_id) REFERENCES olist_customers(customer_id)
);
```

#### Table: olist_customers
```sql
CREATE TABLE olist_customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    customer_unique_id VARCHAR(50) NOT NULL,
    customer_zip_code_prefix INTEGER,
    customer_city VARCHAR(50),
    customer_state VARCHAR(2),
    INDEX idx_unique_id (customer_unique_id),
    INDEX idx_state (customer_state)
);
```

#### Table: customer_rfm_data
```sql
CREATE TABLE customer_rfm_data (
    customer_unique_id VARCHAR(50) PRIMARY KEY,
    Recency INTEGER NOT NULL,
    Frequency INTEGER NOT NULL,
    Monetary DECIMAL(10,2) NOT NULL,
    churned INTEGER NOT NULL,
    CHECK (churned IN (0, 1)),
    CHECK (Recency >= 0),
    CHECK (Frequency > 0),
    CHECK (Monetary >= 0)
);
```

---

## 5. Data Relationships

### Entity Relationship Diagram

```
customers (customer_id) ──1:N──> orders (customer_id)
orders (order_id) ──1:N──> order_items (order_id)
orders (order_id) ──1:N──> order_payments (order_id)
orders (order_id) ──1:1──> order_reviews (order_id)
order_items (product_id) ──N:1──> products (product_id)
order_items (seller_id) ──N:1──> sellers (seller_id)
products (product_category_name) ──N:1──> category_translation (product_category_name)
customers (zip_code_prefix) ──N:1──> geolocation (zip_code_prefix)
sellers (zip_code_prefix) ──N:1──> geolocation (zip_code_prefix)
```

### Key Relationships

1. **Customer → Orders**: One-to-Many
   - A customer can have multiple orders
   - Join key: `customer_id`

2. **Order → Order Items**: One-to-Many
   - An order can contain multiple products
   - Join key: `order_id`

3. **Order → Payments**: One-to-Many
   - An order can have multiple payment methods
   - Join key: `order_id`

4. **Order → Reviews**: One-to-One
   - Each order has at most one review
   - Join key: `order_id`

---

## 6. Data Quality Rules

### 6.1 Validation Rules

#### Orders Table
```python
# Rule 1: order_id must be unique
assert df['order_id'].is_unique

# Rule 2: purchase timestamp must be before delivery
assert (df['order_purchase_timestamp'] <= df['order_delivered_customer_date']).all()

# Rule 3: status must be valid
valid_statuses = ['delivered', 'shipped', 'canceled', 'unavailable', 'invoiced', 'processing']
assert df['order_status'].isin(valid_statuses).all()
```

#### RFM Table
```python
# Rule 1: Recency >= 0
assert (df['Recency'] >= 0).all()

# Rule 2: Frequency > 0
assert (df['Frequency'] > 0).all()

# Rule 3: Monetary >= 0
assert (df['Monetary'] >= 0).all()

# Rule 4: churned in {0, 1}
assert df['churned'].isin([0, 1]).all()
```

### 6.2 Data Cleaning Steps

1. **Remove Duplicates**
   ```python
   df = df.drop_duplicates(subset=['order_id'])
   ```

2. **Handle Missing Values**
   ```python
   # Drop rows with missing critical fields
   df = df.dropna(subset=['order_id', 'customer_id', 'order_purchase_timestamp'])

   # Fill non-critical fields
   df['order_approved_at'].fillna(df['order_purchase_timestamp'], inplace=True)
   ```

3. **Fix Data Types**
   ```python
   df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
   df['payment_value'] = df['payment_value'].astype(float)
   ```

4. **Remove Outliers**
   ```python
   # Remove orders with unrealistic payment values
   df = df[df['payment_value'] < 10000]  # Max 10,000 BRL

   # Remove invalid delivery times
   df = df[df['delivery_days'] <= 365]
   ```

---

## Appendix

### A. Brazilian States Reference

| Code | State | Region |
|------|-------|--------|
| SP | São Paulo | Southeast |
| RJ | Rio de Janeiro | Southeast |
| MG | Minas Gerais | Southeast |
| RS | Rio Grande do Sul | South |
| PR | Paraná | South |
| SC | Santa Catarina | South |
| BA | Bahia | Northeast |
| DF | Federal District | Central-West |

### B. Currency Reference

- **Currency**: Brazilian Real (BRL / R$)
- **Conversion** (approx.): 1 USD ≈ 5 BRL (2018 average)
- All monetary values in dataset are in BRL

### C. Date Formats

- **Timestamp Format**: YYYY-MM-DD HH:MM:SS
- **Date Format**: YYYY-MM-DD
- **Timezone**: UTC-3 (Brazil)

---

**For questions about data definitions, contact the data engineering team or refer to the original Kaggle dataset documentation.**
