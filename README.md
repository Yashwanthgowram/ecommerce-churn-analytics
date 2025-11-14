# E-Commerce Analytics: Full-Stack BI & ML Platform

A comprehensive end-to-end data analytics solution combining business intelligence dashboards with machine learning-powered customer churn prediction. This project analyzes 100,000+ e-commerce transactions to identify customer retention opportunities and predict future behavior.

---

##  Live Demo

**Try the interactive churn prediction app:**  
**[Live Streamlit App](https://ecommerce-churn-analytics-u45usylntzbncyugay95mn.streamlit.app/)**

---

## 📊 Project Overview

This full-stack analytics platform answers two critical business questions:
1. **What happened?** (Business Intelligence Dashboard)
2. **What will happen?** (Predictive ML Application)

### Key Achievements
- 🎯 Identified **$8M customer retention opportunity** (40K at-risk customers)
- 📉 Uncovered critical **3.1% repeat purchase rate** requiring immediate strategy
- 🤖 Built ML model with **85%+ accuracy** for real-time churn predictions
- 📊 Created executive dashboard with **8 KPIs** and **5 interactive visualizations**

---

## 🏗️ Architecture
```
📦 Raw Data (9 CSV files - 100K+ transactions)
    ↓
🐍 Python ETL Pipeline
    ├─ merge_data.py (Data Integration)
    ├─ explore_clean.py (Feature Engineering)
    └─ train_models.py (ML Training)
    ↓
🗄️ PostgreSQL Database
    ├─ olist_orders
    ├─ olist_customers
    └─ customer_rfm_data
    ↓
    ├──→ 📊 Metabase BI Dashboard
    │    • Revenue Analysis
    │    • Geographic Insights
    │    • RFM Segmentation
    │
    └──→ 🤖 ML Models (Scikit-learn + Prophet)
         • Random Forest Churn Classifier
         • Sales Forecasting
         ↓
         🎯 Streamlit Web App (Deployed)
         • Real-time Predictions
         • Customer Benchmarking
         • Actionable Recommendations
```

---

## 📸 Screenshots

### Business Intelligence Dashboard (Metabase)
<img width="853" height="1400" alt="image" src="https://github.com/user-attachments/assets/b9c05b4c-6334-4a8d-b357-83c65d67c821" />


**Key Features:**
- 8 Executive KPIs (Total Revenue, AOV, Customer Count, etc.)
- Monthly Revenue Trends
- Top 10 Product Categories
- RFM Customer Segmentation
- Geographic Sales Analysis
- Interactive filters by State & Payment Type

---

### Predictive Churn Application (Streamlit)

#### Main Interface
<img width="1456" height="819" alt="image" src="https://github.com/user-attachments/assets/68fc1cc6-4c2e-4d2c-ac5c-4dd09d7343c0" />

#### High Risk Prediction
<img width="1512" height="807" alt="image" src="https://github.com/user-attachments/assets/07c9fa4b-810e-4edc-8b9f-bb353f4b8b9d" />

#### Low Risk Prediction
<img width="1512" height="803" alt="image" src="https://github.com/user-attachments/assets/4d75c214-40eb-49e7-b3d8-aa9d3434405e" />

#### Customer Segment Analysis
<img width="1568" height="738" alt="image" src="https://github.com/user-attachments/assets/7553061e-590f-43fd-820a-518e8adb2932" />

---

## 💻 Tech Stack

### Data Engineering & ML
- **Python 3.x** - Core programming language
- **Pandas** - Data manipulation and ETL
- **Scikit-learn** - Machine learning (Random Forest)
- **Prophet** - Time series forecasting
- **Joblib** - Model serialization

### Database & BI
- **PostgreSQL** - Relational database
- **SQLAlchemy** - Database ORM
- **pgAdmin 4** - Database management
- **Metabase** (Docker) - BI dashboards

### Web Application
- **Streamlit** - Web framework
- **Plotly** - Interactive visualizations
- **Streamlit Cloud** - Deployment platform

---

## 🎯 Key Features

### 1. Business Intelligence Dashboard
- **8 Executive KPIs** tracked in real-time
- **RFM Customer Segmentation** (Champions, Loyal, At Risk, Lost)
- **Geographic Analysis** across Brazilian states
- **Revenue Trends** with month-over-month comparison
- **Category Performance** drill-downs
- **Interactive Filters** for dynamic analysis

### 2. Churn Prediction Application
- **Real-time ML Predictions** using Random Forest
- **Risk Scoring** with visual gauge (0-100%)
- **Customer Benchmarking** against database
- **Segment Classification** (6 customer types)
- **Actionable Recommendations** for retention
- **Distribution Analysis** (Recency, Frequency, Monetary)

---



---

## 📊 Data Source

**Olist E-Commerce Dataset**  
- Source: [Kaggle](https://www.kaggle.com/olistbr/brazilian-ecommerce)
- Size: 100,000+ orders from 2016-2018
- Geography: Brazil
- Files: 9 CSV files (orders, customers, products, payments, reviews, etc.)

---
