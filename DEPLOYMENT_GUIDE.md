# Deployment Guide - E-Commerce Churn Analytics Platform

**Version**: 1.0
**Last Updated**: 2025-11-24
**Target Environments**: Local, Cloud (AWS/Azure/GCP)

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Database Configuration](#database-configuration)
4. [Running the ETL Pipeline](#running-the-etl-pipeline)
5. [Launching Applications](#launching-applications)
6. [Cloud Deployment](#cloud-deployment)
7. [Docker Deployment](#docker-deployment)
8. [Troubleshooting](#troubleshooting)
9. [Maintenance](#maintenance)

---

## 1. Prerequisites

### 1.1 System Requirements

**Minimum**:
- OS: Windows 10, macOS 10.14+, Ubuntu 18.04+
- CPU: 2 cores
- RAM: 4 GB
- Storage: 5 GB free space

**Recommended**:
- CPU: 4+ cores
- RAM: 8 GB
- Storage: 10 GB free space
- SSD for faster data processing

### 1.2 Software Dependencies

| Software | Version | Download Link |
|----------|---------|---------------|
| Python | 3.8+ | [python.org](https://www.python.org/downloads/) |
| PostgreSQL | 13+ | [postgresql.org](https://www.postgresql.org/download/) |
| Java JDK | 11+ | [oracle.com](https://www.oracle.com/java/technologies/downloads/) |
| Git | Latest | [git-scm.com](https://git-scm.com/downloads/) |

**Optional**:
- Docker Desktop (for containerized deployment)
- pgAdmin 4 (PostgreSQL GUI)
- VS Code or PyCharm (code editor)

---

## 2. Local Development Setup

### 2.1 Step-by-Step Installation

#### Step 1: Clone Repository
```bash
# Clone from GitHub
git clone https://github.com/Yashwanthgowram/ecommerce-churn-analytics.git
cd ecommerce-churn-analytics
```

#### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

#### Step 3: Install Python Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Verify installation
pip list
```

#### Step 4: Download Dataset
1. Visit [Kaggle Olist Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
2. Click "Download" (requires Kaggle account)
3. Extract `archive.zip`
4. Copy all 9 CSV files to project root directory

**Verify files**:
```bash
# Windows:
dir *.csv

# macOS/Linux:
ls -lh *.csv

# Expected files:
# olist_orders_dataset.csv
# olist_customers_dataset.csv
# olist_order_items_dataset.csv
# olist_order_payments_dataset.csv
# olist_order_reviews_dataset.csv
# olist_products_dataset.csv
# olist_sellers_dataset.csv
# olist_geolocation_dataset.csv
# product_category_name_translation.csv
```

---

## 3. Database Configuration

### 3.1 PostgreSQL Installation

#### Windows
1. Download installer from postgresql.org
2. Run installer (default settings recommended)
3. Remember the password for `postgres` user
4. Add PostgreSQL to PATH:
   - System Properties → Environment Variables
   - Add `C:\Program Files\PostgreSQL\13\bin` to PATH

#### macOS
```bash
# Install via Homebrew
brew install postgresql@13

# Start PostgreSQL service
brew services start postgresql@13
```

#### Linux (Ubuntu)
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start service
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### 3.2 Database Setup

```bash
# Connect to PostgreSQL
psql -U postgres

# In PostgreSQL shell:
```

```sql
-- Create database
CREATE DATABASE ecommerce_analytics;

-- Create dedicated user (optional but recommended)
CREATE USER ecommerce_user WITH PASSWORD 'your_secure_password_here';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE ecommerce_analytics TO ecommerce_user;

-- Exit
\q
```

### 3.3 Configure Database Connection

Edit `load_to_postgres.py` and update the connection string:

```python
# Option 1: Using default postgres user
DATABASE_URL = "postgresql://postgres:your_password@localhost:5432/ecommerce_analytics"

# Option 2: Using dedicated user
DATABASE_URL = "postgresql://ecommerce_user:your_password@localhost:5432/ecommerce_analytics"
```

**Test connection**:
```bash
# Try connecting
psql -U postgres -d ecommerce_analytics

# Should see:
# ecommerce_analytics=#
```

---

## 4. Running the ETL Pipeline

### 4.1 Pipeline Execution Order

**IMPORTANT**: Scripts must be run in this exact order!

```
1. merge_data.py          → Creates master dataset
2. explore_and_clean.py   → Generates RFM data
3. train_models.py        → Trains ML models
4. load_to_postgres.py    → Loads data to database
```

### 4.2 Step-by-Step Execution

#### Step 1: Merge Datasets
```bash
python merge_data.py
```

**Expected Output**:
```
Starting data merge...
All 9 files loaded into memory.
All files successfully merged.
SUCCESS: Master dataset saved as 'olist_master_dataset.csv'
It has 120653 rows and 54 columns.
```

**Verification**:
```bash
# Check file size (should be ~500 MB)
# Windows:
dir olist_master_dataset.csv

# macOS/Linux:
ls -lh olist_master_dataset.csv
```

#### Step 2: Feature Engineering
```bash
python explore_and_clean.py
```

**Expected Output**:
```
Loading master dataset...
Calculating RFM metrics...
RFM data saved to 'customer_rfm_data.csv'
Creating time-series data...
Daily sales data saved to 'daily_sales_forecast_data.csv'
Done!
```

**Verification**:
```python
# Quick check
import pandas as pd
rfm = pd.read_csv('customer_rfm_data.csv')
print(rfm.head())
print(f"Shape: {rfm.shape}")
# Expected: ~96k rows, 4 columns (customer_unique_id, Recency, Frequency, Monetary, churned)
```

#### Step 3: Train Models
```bash
python train_models.py
```

**Expected Output**:
```
Training churn prediction model...
Train/Test Split: 80/20
Training Random Forest Classifier...
Model trained successfully!
Accuracy: 0.852
Precision: 0.821
Recall: 0.883
F1-Score: 0.851
Model saved to 'churn_model.pkl'

Training sales forecasting model...
(If Prophet installed)
Forecast saved to 'sales_forecast_output.csv'
```

**Verification**:
```bash
# Check model file exists
# Windows:
dir churn_model.pkl

# macOS/Linux:
ls -lh churn_model.pkl
# Expected size: ~5 MB
```

#### Step 4: Load to Database
```bash
python load_to_postgres.py
```

**Expected Output**:
```
Connecting to PostgreSQL...
Connection successful!
Loading olist_orders...
Loaded 99441 rows
Loading olist_customers...
Loaded 99441 rows
Loading customer_rfm_data...
Loaded 96096 rows
Creating indexes...
Done! All data loaded successfully.
```

**Verification**:
```sql
-- In PostgreSQL shell:
psql -U postgres -d ecommerce_analytics

-- Check tables
\dt

-- Check row counts
SELECT COUNT(*) FROM olist_orders;
SELECT COUNT(*) FROM customer_rfm_data;

-- Sample query
SELECT * FROM customer_rfm_data LIMIT 5;
```

---

## 5. Launching Applications

### 5.1 Streamlit Web Application

#### Launch Command
```bash
streamlit run app.py
```

**Expected Output**:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.10:8501
```

#### Access Application
1. Open browser
2. Navigate to `http://localhost:8501`
3. Should see "Customer Churn Intelligence Platform"

#### Custom Port (if 8501 is busy)
```bash
streamlit run app.py --server.port 8502
```

#### Run in Background
```bash
# macOS/Linux:
nohup streamlit run app.py &

# Windows (using PowerShell):
Start-Process streamlit run app.py
```

### 5.2 Metabase BI Dashboard

#### Launch Metabase
```bash
# Navigate to project directory
cd /path/to/project

# Launch Metabase
java -jar metabase.jar
```

**Expected Output**:
```
2025-11-24 10:00:00 INFO metabase.core :: Starting Metabase...
2025-11-24 10:00:05 INFO metabase.core :: Metabase Initialization COMPLETE
2025-11-24 10:00:06 INFO metabase.core :: Metabase is available at http://localhost:3000
```

#### Initial Setup (First Time Only)
1. Open browser at `http://localhost:3000`
2. Click "Let's get started"
3. Create admin account:
   - Email: your_email@example.com
   - Password: (strong password)
4. Add database connection:
   - Database type: PostgreSQL
   - Host: localhost
   - Port: 5432
   - Database name: ecommerce_analytics
   - Username: postgres (or ecommerce_user)
   - Password: (your password)
5. Click "Connect database"
6. Let Metabase scan the schema (2-3 minutes)

#### Create First Dashboard
1. Click "New" → "Dashboard"
2. Name: "E-Commerce Executive Dashboard"
3. Add questions:
   - Total Revenue: `SUM(payment_value)`
   - Total Orders: `COUNT(DISTINCT order_id)`
   - Churn Rate: `AVG(churned) * 100`
   - Top States: `GROUP BY customer_state`

---

## 6. Cloud Deployment

### 6.1 AWS Deployment

#### Architecture
```
AWS Cloud
├── EC2 Instance (Streamlit app)
├── RDS PostgreSQL (Database)
├── S3 Bucket (Data files, models)
└── Elastic Load Balancer (Optional)
```

#### Step-by-Step AWS Deployment

##### 1. Launch EC2 Instance
```bash
# Specifications:
- AMI: Ubuntu 20.04 LTS
- Instance Type: t3.medium (2 vCPU, 4 GB RAM)
- Security Group: Allow ports 22, 8501, 3000
```

##### 2. Setup RDS PostgreSQL
```bash
# RDS Configuration:
- Engine: PostgreSQL 13
- Instance Class: db.t3.micro (free tier eligible)
- Storage: 20 GB
- Public accessibility: Yes (for development)
- Security Group: Allow port 5432 from EC2
```

##### 3. Connect to EC2
```bash
ssh -i your-key.pem ubuntu@ec2-public-ip
```

##### 4. Install Dependencies on EC2
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.8+
sudo apt install python3.8 python3-pip -y

# Install PostgreSQL client
sudo apt install postgresql-client -y

# Install Git
sudo apt install git -y
```

##### 5. Deploy Application
```bash
# Clone repository
git clone https://github.com/Yashwanthgowram/ecommerce-churn-analytics.git
cd ecommerce-churn-analytics

# Install Python packages
pip3 install -r requirements.txt

# Update database connection to RDS endpoint
nano load_to_postgres.py
# Replace localhost with RDS endpoint:
# DATABASE_URL = "postgresql://user:pass@your-rds-endpoint.rds.amazonaws.com:5432/ecommerce_analytics"

# Run ETL pipeline
python3 merge_data.py
python3 explore_and_clean.py
python3 train_models.py
python3 load_to_postgres.py
```

##### 6. Run Streamlit with Process Manager
```bash
# Install screen (process manager)
sudo apt install screen -y

# Start new screen session
screen -S streamlit

# Run Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Detach from screen: Ctrl+A, then D
# Reattach: screen -r streamlit
```

##### 7. Configure Nginx (Optional - Production)
```bash
# Install Nginx
sudo apt install nginx -y

# Create config file
sudo nano /etc/nginx/sites-available/streamlit

# Add configuration:
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 6.2 Azure Deployment

Similar steps using:
- Azure VM (instead of EC2)
- Azure Database for PostgreSQL (instead of RDS)
- Azure Blob Storage (instead of S3)

---

## 7. Docker Deployment

### 7.1 Dockerfile

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 7.2 Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: ecommerce_analytics
      POSTGRES_USER: ecommerce_user
      POSTGRES_PASSWORD: your_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  streamlit:
    build: .
    ports:
      - "8501:8501"
    depends_on:
      - postgres
    environment:
      DATABASE_URL: postgresql://ecommerce_user:your_password@postgres:5432/ecommerce_analytics

  metabase:
    image: metabase/metabase:latest
    ports:
      - "3000:3000"
    depends_on:
      - postgres
    environment:
      MB_DB_TYPE: postgres
      MB_DB_DBNAME: ecommerce_analytics
      MB_DB_PORT: 5432
      MB_DB_USER: ecommerce_user
      MB_DB_PASS: your_password
      MB_DB_HOST: postgres

volumes:
  postgres_data:
```

### 7.3 Run with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f streamlit

# Stop all services
docker-compose down
```

---

## 8. Troubleshooting

### 8.1 Common Issues

#### Issue: "Module not found"
```bash
# Solution: Reinstall packages
pip install -r requirements.txt --force-reinstall
```

#### Issue: "Database connection failed"
```bash
# Solution 1: Check PostgreSQL is running
# Windows:
sc query postgresql-x64-13

# macOS:
brew services list

# Linux:
sudo systemctl status postgresql

# Solution 2: Test connection manually
psql -U postgres -d ecommerce_analytics

# Solution 3: Check firewall/port
telnet localhost 5432
```

#### Issue: "CSV files not found"
```bash
# Solution: Verify files in project directory
ls -la *.csv

# Download again if missing
```

#### Issue: "Streamlit port already in use"
```bash
# Find process using port 8501
# Windows:
netstat -ano | findstr :8501

# macOS/Linux:
lsof -i :8501

# Kill process or use different port
streamlit run app.py --server.port 8502
```

#### Issue: "Model file not found" in Streamlit
```bash
# Solution: Run training script
python train_models.py

# Verify model file
ls -lh churn_model.pkl
```

### 8.2 Logs and Debugging

#### Enable Streamlit Debug Mode
```bash
streamlit run app.py --logger.level=debug
```

#### Check PostgreSQL Logs
```bash
# Ubuntu:
sudo tail -f /var/log/postgresql/postgresql-13-main.log

# macOS (Homebrew):
tail -f /usr/local/var/log/postgresql@13.log
```

---

## 9. Maintenance

### 9.1 Regular Updates

#### Update Dependencies (Monthly)
```bash
pip list --outdated
pip install --upgrade package_name
pip freeze > requirements.txt
```

#### Database Backups (Weekly)
```bash
# Create backup
pg_dump -U postgres ecommerce_analytics > backup_$(date +%Y%m%d).sql

# Restore backup
psql -U postgres ecommerce_analytics < backup_20251124.sql
```

#### Model Retraining (Monthly)
```bash
# Download latest data
# Re-run pipeline
python merge_data.py
python explore_and_clean.py
python train_models.py

# Restart Streamlit to load new model
```

### 9.2 Monitoring

#### Monitor Application Health
```bash
# Check Streamlit process
ps aux | grep streamlit

# Monitor resource usage
top
# or
htop
```

#### Monitor Database Performance
```sql
-- Active connections
SELECT count(*) FROM pg_stat_activity;

-- Table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

---

## Appendix

### A. Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] PostgreSQL 13+ installed and running
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] 9 CSV files downloaded and placed in project root
- [ ] Database created (`ecommerce_analytics`)
- [ ] Pipeline scripts run in order (merge → clean → train → load)
- [ ] Streamlit app launched (`streamlit run app.py`)
- [ ] Application accessible at `http://localhost:8501`

### B. Support Resources

- **Documentation**: See DOCUMENTATION.md
- **Architecture**: See ARCHITECTURE.md
- **GitHub Issues**: https://github.com/Yashwanthgowram/ecommerce-churn-analytics/issues
- **Streamlit Docs**: https://docs.streamlit.io/
- **PostgreSQL Docs**: https://www.postgresql.org/docs/

---

**Deployment complete! Your E-Commerce Churn Analytics Platform is now running.**
