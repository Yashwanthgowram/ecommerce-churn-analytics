# Technical Architecture - E-Commerce Churn Analytics Platform

**Version**: 1.0
**Last Updated**: 2025-11-24
**Architecture Type**: Batch Processing + Real-time Serving

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Technology Stack](#technology-stack)
5. [Machine Learning Pipeline](#machine-learning-pipeline)
6. [Application Architecture](#application-architecture)
7. [Database Design](#database-design)
8. [Deployment Architecture](#deployment-architecture)
9. [Security Architecture](#security-architecture)
10. [Performance Considerations](#performance-considerations)

---

## 1. System Architecture Overview

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCE LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  Kaggle Olist Dataset (9 CSV Files)                             │
│  • orders.csv        • customers.csv      • products.csv        │
│  • order_items.csv   • payments.csv       • reviews.csv         │
│  • sellers.csv       • geolocation.csv    • categories.csv      │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ETL / DATA PROCESSING LAYER                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │  merge_data.py   │  │ explore_clean.py │  │ train_models  │ │
│  │                  │  │                  │  │      .py      │ │
│  │ • Join 9 files   │→ │ • RFM Analysis   │→ │ • Random      │ │
│  │ • Data cleaning  │  │ • Feature Eng    │  │   Forest      │ │
│  │ • Validation     │  │ • Time-series    │  │ • Prophet     │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
│             │                    │                    │          │
│             ▼                    ▼                    ▼          │
│  olist_master_dataset.csv  customer_rfm_data.csv  models.pkl   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA WAREHOUSE LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│                     PostgreSQL Database                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ olist_orders │  │olist_customers│ │customer_rfm_data    │  │
│  │              │  │               │ │                      │  │
│  │ • Fact table │  │ • Dimension   │ │ • Aggregated         │  │
│  │ • 100K rows  │  │ • Customer ID │ │ • Analytics-ready    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│  load_to_postgres.py (SQLAlchemy + psycopg2)                    │
└────────┬───────────────────────────────┬────────────────────────┘
         │                               │
         ▼                               ▼
┌────────────────────────┐    ┌───────────────────────────────────┐
│    BI / ANALYTICS      │    │   MACHINE LEARNING SERVING        │
│        LAYER           │    │           LAYER                   │
├────────────────────────┤    ├───────────────────────────────────┤
│   Metabase (Docker)    │    │    Streamlit Web Application      │
│                        │    │                                   │
│ • SQL Queries          │    │  ┌────────────────────────────┐  │
│ • Interactive Dashboards│   │  │  app.py                    │  │
│ • Visualizations       │    │  │  • Load churn_model.pkl    │  │
│ • Filters & Drill-down │    │  │  • RFM Input Interface     │  │
│ • 8 Executive KPIs     │    │  │  • Real-time Predictions   │  │
│ • Revenue Trends       │    │  │  • Plotly Visualizations   │  │
│ • Geographic Analysis  │    │  │  • Retention Strategies    │  │
│                        │    │  └────────────────────────────┘  │
│ Port: 3000             │    │  Port: 8501                       │
└────────────────────────┘    └───────────────────────────────────┘
         │                               │
         ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       END USERS / CONSUMERS                      │
├─────────────────────────────────────────────────────────────────┤
│  • Business Analysts        • Marketing Teams                   │
│  • Data Scientists          • Customer Success Managers         │
│  • Product Managers         • Executives                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Architecture

### 2.1 Data Ingestion Component

**File**: `merge_data.py`

```python
┌────────────────────────────────────┐
│    merge_data.py                   │
├────────────────────────────────────┤
│                                    │
│  Input:                            │
│    └─ 9 CSV files (Kaggle)        │
│                                    │
│  Processing:                       │
│    ├─ Load CSVs with pandas       │
│    ├─ Perform LEFT JOINs          │
│    ├─ Handle one-to-many relations │
│    └─ Validate merged data        │
│                                    │
│  Output:                           │
│    └─ olist_master_dataset.csv    │
│       (500MB, 120K rows)           │
│                                    │
│  Dependencies:                     │
│    └─ pandas, os                   │
└────────────────────────────────────┘
```

### 2.2 Feature Engineering Component

**File**: `explore_and_clean.py`

```python
┌────────────────────────────────────┐
│   explore_and_clean.py             │
├────────────────────────────────────┤
│                                    │
│  Input:                            │
│    └─ olist_master_dataset.csv    │
│                                    │
│  Processing:                       │
│    ├─ RFM Calculation              │
│    │  ├─ Recency (days)            │
│    │  ├─ Frequency (orders)        │
│    │  └─ Monetary (total value)    │
│    │                                │
│    ├─ Churn Labeling               │
│    │  └─ churned = 1 if R > 180   │
│    │                                │
│    ├─ Time-series Aggregation      │
│    │  └─ Daily sales totals        │
│    │                                │
│    └─ Data Quality Checks          │
│                                    │
│  Outputs:                          │
│    ├─ customer_rfm_data.csv        │
│    └─ daily_sales_forecast_data.csv│
│                                    │
│  Dependencies:                     │
│    └─ pandas, numpy, datetime      │
└────────────────────────────────────┘
```

### 2.3 Model Training Component

**File**: `train_models.py`

```python
┌────────────────────────────────────┐
│      train_models.py               │
├────────────────────────────────────┤
│                                    │
│  Input:                            │
│    ├─ customer_rfm_data.csv        │
│    └─ daily_sales_forecast_data.csv│
│                                    │
│  Model 1: Churn Prediction         │
│    ├─ Algorithm: Random Forest     │
│    ├─ Features: R, F, M            │
│    ├─ Target: churned (binary)    │
│    ├─ Train/Test Split: 80/20     │
│    └─ Evaluation: Accuracy, F1    │
│                                    │
│  Model 2: Sales Forecasting        │
│    ├─ Algorithm: Prophet           │
│    ├─ Input: ds, y (date, sales)  │
│    ├─ Horizon: 180 days            │
│    └─ Output: yhat with CI         │
│                                    │
│  Outputs:                          │
│    ├─ churn_model.pkl              │
│    └─ sales_forecast_output.csv    │
│                                    │
│  Dependencies:                     │
│    ├─ scikit-learn                 │
│    ├─ joblib                       │
│    └─ prophet (optional)           │
└────────────────────────────────────┘
```

### 2.4 Data Loading Component

**File**: `load_to_postgres.py`

```python
┌────────────────────────────────────┐
│    load_to_postgres.py             │
├────────────────────────────────────┤
│                                    │
│  Input:                            │
│    ├─ olist_master_dataset.csv    │
│    └─ customer_rfm_data.csv        │
│                                    │
│  Processing:                       │
│    ├─ Establish DB connection      │
│    │  └─ SQLAlchemy engine         │
│    │                                │
│    ├─ Create tables (if not exist) │
│    │                                │
│    ├─ Load data via bulk insert    │
│    │  └─ to_sql() method           │
│    │                                │
│    └─ Create indexes               │
│                                    │
│  Output:                           │
│    └─ PostgreSQL tables populated  │
│                                    │
│  Dependencies:                     │
│    ├─ sqlalchemy                   │
│    ├─ psycopg2-binary              │
│    └─ pandas                       │
└────────────────────────────────────┘
```

### 2.5 Web Application Component

**File**: `app.py`

```python
┌────────────────────────────────────┐
│         app.py (Streamlit)         │
├────────────────────────────────────┤
│                                    │
│  Initialization:                   │
│    ├─ Load model (joblib)          │
│    ├─ Load RFM data (cache)        │
│    └─ Configure UI (CSS)           │
│                                    │
│  UI Components:                    │
│    ├─ Sidebar                      │
│    │  ├─ Customer ID input         │
│    │  ├─ Recency selector          │
│    │  ├─ Frequency selector        │
│    │  ├─ Monetary selector         │
│    │  └─ Predict button            │
│    │                                │
│    ├─ Main Area                    │
│    │  ├─ KPI metrics (4 cards)     │
│    │  ├─ Prediction result         │
│    │  ├─ Risk gauge (Plotly)       │
│    │  ├─ Retention strategies      │
│    │  ├─ Benchmarking histograms   │
│    │  └─ Segment pie chart         │
│    │                                │
│    └─ Footer                       │
│                                    │
│  Business Logic:                   │
│    ├─ Segment classification       │
│    ├─ Churn probability calc       │
│    ├─ Percentile calculations      │
│    └─ Strategy recommendations     │
│                                    │
│  Dependencies:                     │
│    ├─ streamlit                    │
│    ├─ plotly                       │
│    ├─ pandas                       │
│    └─ joblib                       │
└────────────────────────────────────┘
```

---

## 3. Data Flow Architecture

### 3.1 Batch Data Pipeline Flow

```
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│  Download Olist CSV files       │
│  from Kaggle                    │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Run: python merge_data.py      │
│  • Load 9 CSV files             │
│  • Perform JOINs                │
│  • Output: master dataset       │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Run: python explore_clean.py   │
│  • Calculate RFM                │
│  • Label churn                  │
│  • Aggregate time-series        │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Run: python train_models.py    │
│  • Train Random Forest          │
│  • Train Prophet (optional)     │
│  • Save models                  │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Run: python load_to_postgres.py│
│  • Connect to PostgreSQL        │
│  • Create tables                │
│  • Bulk insert data             │
└──────┬──────────────────────────┘
       │
       ▼
┌──────────────┬──────────────────┐
│  READY FOR   │   READY FOR      │
│  BI ANALYSIS │   PREDICTIONS    │
│  (Metabase)  │   (Streamlit)    │
└──────────────┴──────────────────┘
```

### 3.2 Real-time Prediction Flow

```
User Inputs RFM Values
       │
       ▼
┌────────────────────────┐
│  Streamlit UI          │
│  • Recency slider      │
│  • Frequency dropdown  │
│  • Monetary dropdown   │
└───────┬────────────────┘
        │
        ▼ (Click "Analyze")
┌────────────────────────┐
│  Create input DataFrame│
│  [[R, F, M]]           │
└───────┬────────────────┘
        │
        ▼
┌────────────────────────┐
│  Load churn_model.pkl  │
│  (cached in memory)    │
└───────┬────────────────┘
        │
        ▼
┌────────────────────────┐
│  model.predict()       │
│  model.predict_proba() │
└───────┬────────────────┘
        │
        ▼
┌────────────────────────┐
│  Calculate:            │
│  • Churn probability   │
│  • Risk level          │
│  • Segment             │
│  • Percentiles         │
└───────┬────────────────┘
        │
        ▼
┌────────────────────────┐
│  Render Results:       │
│  • Prediction box      │
│  • Gauge chart         │
│  • Histograms          │
│  • Recommendations     │
└────────────────────────┘
```

---

## 4. Technology Stack

### 4.1 Core Technologies

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Language** | Python | 3.8+ | Primary programming language |
| **Data Processing** | Pandas | 1.3+ | ETL, data manipulation |
| **ML Framework** | scikit-learn | 1.0+ | Churn prediction model |
| **Forecasting** | Prophet | 1.1+ | Sales forecasting (optional) |
| **Web Framework** | Streamlit | 1.28+ | Interactive UI |
| **Visualization** | Plotly | 5.0+ | Dynamic charts |
| **Database** | PostgreSQL | 13+ | Data warehouse |
| **DB Driver** | psycopg2 | 2.9+ | PostgreSQL adapter |
| **ORM** | SQLAlchemy | 1.4+ | Database abstraction |
| **Serialization** | joblib | 1.1+ | Model persistence |
| **BI Tool** | Metabase | Latest | Business intelligence |
| **Container** | Docker | 20+ | Metabase containerization |

### 4.2 Python Package Dependencies

```requirements.txt
pandas>=1.3.0
scikit-learn>=1.0.0
streamlit>=1.28.0
joblib>=1.1.0
plotly>=5.0.0
psycopg2-binary>=2.9.0
sqlalchemy>=1.4.0
numpy>=1.21.0
prophet>=1.1.0 (optional)
```

---

## 5. Machine Learning Pipeline

### 5.1 Pipeline Architecture

```
┌───────────────────────────────────────────────────────────────┐
│               MACHINE LEARNING PIPELINE                       │
└───────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  DATA        │───>│  FEATURE     │───>│  MODEL       │
│  COLLECTION  │    │  ENGINEERING │    │  TRAINING    │
└──────────────┘    └──────────────┘    └──────────────┘
                                               │
                    ┌──────────────┐           │
                    │  MODEL       │<──────────┘
                    │  EVALUATION  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  MODEL       │
                    │  DEPLOYMENT  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  PREDICTION  │
                    │  SERVING     │
                    └──────────────┘
```

### 5.2 Model Training Workflow

```python
# Pseudocode for train_models.py

# 1. Load Data
df = pd.read_csv('customer_rfm_data.csv')

# 2. Prepare Features & Target
X = df[['Recency', 'Frequency', 'Monetary']]
y = df['churned']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize Model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

# 5. Train
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# 7. Serialize Model
joblib.dump(model, 'churn_model.pkl')
```

---

## 6. Application Architecture

### 6.1 Streamlit Application Flow

```
app.py Execution Flow:

1. Page Configuration
   └─ set_page_config()

2. Load Resources (Cached)
   ├─ @st.cache_resource: Load model
   └─ @st.cache_data: Load RFM data

3. Render Sidebar
   ├─ Customer ID input
   ├─ RFM selectors
   └─ Predict button

4. Display Metrics
   └─ 4 KPI cards (Recency, Frequency, Monetary, Segment)

5. Prediction Logic (if button clicked)
   ├─ Create input DataFrame
   ├─ Call model.predict_proba()
   ├─ Calculate metrics
   └─ Store results

6. Visualizations
   ├─ Risk gauge (Plotly)
   ├─ Benchmarking histograms
   └─ Segment pie chart

7. Recommendations
   └─ Display retention strategies

8. Footer
   └─ Branding & credits
```

### 6.2 State Management

```python
# Streamlit caching strategy

@st.cache_resource  # Cached globally, never expires
def load_model():
    return joblib.load('churn_model.pkl')

@st.cache_data  # Cached per session, TTL configurable
def load_rfm_data():
    return pd.read_csv('customer_rfm_data.csv')

# Session state (if needed)
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
```

---

## 7. Database Design

### 7.1 Schema Diagram

```sql
┌──────────────────────┐
│   olist_orders       │
├──────────────────────┤
│ PK order_id          │
│ FK customer_id       │─────────┐
│    order_status      │         │
│    order_purchase_ts │         │
│    ...               │         │
└──────────────────────┘         │
                                 │
┌──────────────────────┐         │
│  olist_customers     │<────────┘
├──────────────────────┤
│ PK customer_id       │
│    customer_unique_id│───┐
│    city              │   │
│    state             │   │
└──────────────────────┘   │
                           │
┌──────────────────────┐   │
│ customer_rfm_data    │<──┘
├──────────────────────┤
│ PK customer_unique_id│
│    Recency           │
│    Frequency         │
│    Monetary          │
│    churned           │
└──────────────────────┘
```

### 7.2 Indexing Strategy

```sql
-- Primary keys (automatically indexed)
-- olist_orders.order_id
-- olist_customers.customer_id
-- customer_rfm_data.customer_unique_id

-- Foreign keys
CREATE INDEX idx_orders_customer ON olist_orders(customer_id);

-- Date filtering
CREATE INDEX idx_orders_date ON olist_orders(order_purchase_timestamp);

-- RFM queries
CREATE INDEX idx_rfm_recency ON customer_rfm_data(Recency);
CREATE INDEX idx_rfm_churned ON customer_rfm_data(churned);

-- Geographic queries
CREATE INDEX idx_customer_state ON olist_customers(customer_state);
```

---

## 8. Deployment Architecture

### 8.1 Local Development Setup

```
Development Machine
├── Python 3.8+ (virtual environment)
├── PostgreSQL 13+ (local instance)
├── Metabase (JAR file execution)
└── Streamlit (local server)

File Structure:
project/
├── *.csv (data files)
├── *.py (scripts)
├── *.pkl (models)
├── metabase.jar
└── requirements.txt
```

### 8.2 Production Deployment (Proposed)

```
┌──────────────────────────────────────┐
│      Cloud Infrastructure            │
├──────────────────────────────────────┤
│                                      │
│  ┌────────────────────────────────┐ │
│  │  Application Server            │ │
│  │  (AWS EC2 / Azure VM)          │ │
│  │                                │ │
│  │  • Streamlit app (port 8501)   │ │
│  │  • Gunicorn/Uvicorn (WSGI)     │ │
│  │  • Nginx (reverse proxy)       │ │
│  └────────────────────────────────┘ │
│                                      │
│  ┌────────────────────────────────┐ │
│  │  Database Server               │ │
│  │  (AWS RDS / Azure Database)    │ │
│  │                                │ │
│  │  • PostgreSQL 13+              │ │
│  │  • Automated backups           │ │
│  │  • Read replicas               │ │
│  └────────────────────────────────┘ │
│                                      │
│  ┌────────────────────────────────┐ │
│  │  BI Server                     │ │
│  │  (Docker Container)            │ │
│  │                                │ │
│  │  • Metabase (port 3000)        │ │
│  │  • Docker Compose              │ │
│  └────────────────────────────────┘ │
│                                      │
│  ┌────────────────────────────────┐ │
│  │  Storage                       │ │
│  │  (S3 / Azure Blob)             │ │
│  │                                │ │
│  │  • CSV files                   │ │
│  │  • Model artifacts             │ │
│  │  • Logs                        │ │
│  └────────────────────────────────┘ │
└──────────────────────────────────────┘
```

---

## 9. Security Architecture

### 9.1 Security Layers

```
┌─────────────────────────────────┐
│  Application Security           │
│  • Input validation             │
│  • SQL injection prevention     │
│  • XSS protection (Streamlit)   │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│  Database Security              │
│  • Password authentication      │
│  • Connection encryption (SSL)  │
│  • Role-based access control    │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│  Network Security               │
│  • Firewall rules               │
│  • VPC/Private networks         │
│  • HTTPS/TLS                    │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│  Data Security                  │
│  • PII anonymization            │
│  • Encryption at rest           │
│  • Backup encryption            │
└─────────────────────────────────┘
```

---

## 10. Performance Considerations

### 10.1 Optimization Strategies

#### Data Processing
- **Chunking**: Process large CSVs in chunks to avoid memory overflow
- **Vectorization**: Use pandas vectorized operations instead of loops
- **Data types**: Optimize dtypes (int8 vs int64, category vs object)

#### Machine Learning
- **Feature selection**: Use only RFM features (minimal overhead)
- **Model complexity**: Random Forest with limited depth (balance accuracy/speed)
- **Caching**: Cache model in memory (Streamlit @st.cache_resource)

#### Database
- **Indexing**: Index frequently queried columns
- **Connection pooling**: Reuse database connections
- **Query optimization**: Use aggregations at DB level, not application level

#### Web Application
- **Lazy loading**: Load data only when needed
- **Caching**: Cache expensive computations (@st.cache_data)
- **Compression**: Enable gzip compression for Streamlit

---

## Appendix

### A. Port Allocations

| Service | Port | Purpose |
|---------|------|---------|
| Streamlit | 8501 | Web application |
| Metabase | 3000 | BI dashboard |
| PostgreSQL | 5432 | Database server |

### B. File Size Estimates

| File | Size | Description |
|------|------|-------------|
| olist_master_dataset.csv | ~500 MB | Merged dataset |
| customer_rfm_data.csv | ~2 MB | RFM analysis |
| churn_model.pkl | ~5 MB | Trained model |
| metabase.jar | ~250 MB | Metabase binary |

---

**For architecture questions, contact the technical lead or refer to the deployment documentation.**
