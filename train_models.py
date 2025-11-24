import pandas as pd
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

print("Starting Step 3: Model Training...")

# --- 1. BUILD FORECAST MODEL ---
print("Training sales forecast model...")
try:
    df_sales = pd.read_csv('daily_sales_forecast_data.csv')
except FileNotFoundError:
    print("Error: 'daily_sales_forecast_data.csv' not found. Please run Step 2 first.")
    exit()

# Initialize Prophet model
model_forecast = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
# Train the model on our daily sales data
model_forecast.fit(df_sales)

# Create a dataframe for future predictions (180 days)
future = model_forecast.make_future_dataframe(periods=180)
# Generate the forecast
forecast = model_forecast.predict(future)

# Save the forecast results to a CSV. We will load this into Power BI.
forecast.to_csv('sales_forecast_output.csv', index=False)
print("SUCCESS: Sales forecast model trained and results saved to 'sales_forecast_output.csv'")


# --- 2. BUILD CHURN MODEL ---
print("Training customer churn model...")
try:
    df_rfm = pd.read_csv('customer_rfm_data.csv')
except FileNotFoundError:
    print("Error: 'customer_rfm_data.csv' not found. Please run Step 2 first.")
    exit()

# Define "churn": If a customer hasn't purchased in 180 days, they are 'churned'
RECENCY_THRESHOLD = 180
df_rfm['is_churned'] = df_rfm['Recency'].apply(lambda x: 1 if x > RECENCY_THRESHOLD else 0)

# 'X' (features) = The data we use to predict (Recency, Frequency, Monetary)
# 'y' (target) = What we WANT to predict (is_churned)
features = ['Recency', 'Frequency', 'Monetary']
target = 'is_churned'
X = df_rfm[features]
y = df_rfm[target]

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
# n_jobs=-1 uses all available CPU cores to speed up training
model_churn = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# Train the model
model_churn.fit(X_train, y_train)

# Test the model's accuracy
y_pred = model_churn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Churn Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model to a file. We will load this into our Streamlit app.
joblib.dump(model_churn, 'churn_model.pkl')
print("SUCCESS: Churn model trained and saved as 'churn_model.pkl'")

print("\nStep 3 Complete!")