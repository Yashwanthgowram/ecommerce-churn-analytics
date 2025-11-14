import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Customer Churn Predictor", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ELEGANT RED CSS ---
st.markdown("""
    <style>
    /* Red gradient background */
    .main {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 50%, #fca5a5 100%);
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 10px 40px rgba(220, 38, 38, 0.15);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: #991b1b;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Prediction boxes */
    .prediction-box {
        padding: 30px;
        border-radius: 16px;
        margin: 20px 0;
        font-size: 20px;
        font-weight: 600;
        text-align: center;
        border: 3px solid;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-color: #dc2626;
        color: #991b1b;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-color: #22c55e;
        color: #166534;
    }
    
    /* Headers */
    h1 {
        color: #991b1b !important;
        font-weight: 800 !important;
        font-size: 42px !important;
        text-align: center;
        margin-bottom: 10px !important;
        text-shadow: 2px 2px 4px rgba(153, 27, 27, 0.1);
    }
    
    h2 {
        color: #b91c1c !important;
        font-weight: 700 !important;
        font-size: 26px !important;
        margin-top: 40px !important;
        margin-bottom: 20px !important;
        padding-bottom: 12px !important;
        border-bottom: 3px solid #ef4444 !important;
    }
    
    h3 {
        color: #dc2626 !important;
        font-weight: 600 !important;
        font-size: 18px !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #dc2626 0%, #ef4444 100%);
    }
    
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* White selectboxes with dark text */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: white !important;
        color: #1f2937 !important;
        border: 1px solid #d1d5db !important; 
        border-radius: 8px !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    [data-testid="stSidebar"] input {
        background-color: white !important;
        color: #1f2937 !important;
        border: 1px solid #d1d5db !important;
    }
    
    /* HIDE THE QUESTION MARK ICONS */
    [data-testid="stSidebar"] button[kind="icon"] {
        display: none !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white !important;
        border: none;
        padding: 14px 28px;
        font-size: 16px;
        font-weight: 700;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.4);
        transition: all 0.3s;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(220, 38, 38, 0.5);
        background: linear-gradient(135deg, #b91c1c 0%, #991b1b 100%);
    }
    
    /* Dividers */
    hr {
        margin: 40px 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #ef4444, transparent);
    }
    
    /* Dataframe styling */
    .dataframe tbody tr:empty {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('churn_model.pkl')
    except FileNotFoundError:
        st.error("Model file not found.")
        st.stop()

model = load_model()

# --- LOAD RFM DATA ---
@st.cache_data
def load_rfm_data():
    try:
        return pd.read_csv('customer_rfm_data.csv')
    except:
        return None

rfm_data = load_rfm_data()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🎯 Customer Profile</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 14px;'>Select customer parameters</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    customer_id = st.text_input(
        "📝 Customer ID",
        value="CUST-" + datetime.now().strftime("%Y%m%d")
    )
    
    st.markdown("### 📊 RFM Parameters")
    
    # Recency dropdown
    recency_options = {
        "0-30 days (Very Recent)": 15,
        "31-60 days (Recent)": 45,
        "61-90 days (Moderate)": 75,
        "91-180 days (Inactive)": 135,
        "181-365 days (Very Inactive)": 270,
        "365+ days (Lost)": 500
    }
    recency_label = st.selectbox(
        "📅 Recency",
        options=list(recency_options.keys()),
        index=3
    )
    recency = recency_options[recency_label]
    
    # Frequency dropdown
    frequency_options = {
        "1 order (New)": 1,
        "2-3 orders (Occasional)": 2,
        "4-6 orders (Regular)": 5,
        "7-10 orders (Frequent)": 8,
        "11-20 orders (Loyal)": 15,
        "20+ orders (Champion)": 25
    }
    frequency_label = st.selectbox(
        "🔄 Frequency",
        options=list(frequency_options.keys()),
        index=1
    )
    frequency = frequency_options[frequency_label]
    
    # Monetary dropdown
    monetary_options = {
        "$0-100 (Low Value)": 50,
        "$101-300 (Standard)": 200,
        "$301-500 (Good Value)": 400,
        "$501-1000 (High Value)": 750,
        "$1001-2500 (Premium)": 1500,
        "$2500+ (VIP)": 4000
    }
    monetary_label = st.selectbox(
        "💰 Monetary",
        options=list(monetary_options.keys()),
        index=1
    )
    monetary = monetary_options[monetary_label]
    
    st.markdown("---")
    predict_button = st.button("🔮 ANALYZE CHURN RISK", use_container_width=True)

# --- SEGMENT CLASSIFICATION ---
def get_segment(r, f, m):
    if r <= 30 and f >= 7 and m >= 1000:
        return "Champions", "🏆"
    elif r <= 60 and f >= 4 and m >= 500:
        return "Loyal Customers", "💎"
    elif r <= 90 and f >= 2 and m >= 300:
        return "Potential Loyalists", "⭐"
    elif r > 180 and f >= 4:
        return "At Risk", "⚠️"
    elif r > 180:
        return "Lost Customers", "❌"
    else:
        return "New Customers", "🆕"

segment, segment_icon = get_segment(recency, frequency, monetary)

# --- HEADER ---
st.markdown("<h1>🎯 Customer Churn Intelligence Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #6b7280; margin-bottom: 30px;'>AI-Powered Predictive Analytics for Customer Retention</p>", unsafe_allow_html=True)

# --- METRICS ---
st.markdown("### 📋 Current Customer Profile")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📅 Recency", f"{recency} days", 
              "Active" if recency < 90 else "Inactive")

with col2:
    st.metric("🔄 Frequency", f"{frequency} orders",
              "High" if frequency >= 7 else "Low" if frequency <= 2 else "Moderate")

with col3:
    st.metric("💰 Monetary", f"${monetary:,}",
              "High" if monetary >= 1000 else "Low" if monetary < 300 else "Standard")

with col4:
    st.metric(f"{segment_icon} Segment", segment)

st.markdown("---")

# --- PREDICTION ---
if predict_button:
    with st.spinner("🔄 Analyzing customer profile..."):
        input_data = pd.DataFrame({
            'Recency': [recency],
            'Frequency': [frequency],
            'Monetary': [monetary]
        })
        
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        churn_prob = prediction_proba[0][1] * 100
        safe_prob = prediction_proba[0][0] * 100
        
        col1, col2 = st.columns([2, 1], gap="large")
        
        with col1:
            if prediction[0] == 1:
                st.markdown(f"""
                    <div class='prediction-box risk-high'>
                        ⚠️ HIGH CHURN RISK DETECTED
                        <div style='font-size: 15px; margin-top: 10px; font-weight: 500;'>
                        Churn Probability: {churn_prob:.1f}% | Customer: {customer_id}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("## 🎯 Recommended Retention Strategy")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("""
                    #### Immediate Actions (Week 1-2)
                    - 📧 Personalized win-back email
                    - 🎁 Exclusive 15-20% discount
                    - 📞 Priority account manager call
                    - 🔍 Analyze purchase behavior
                    - 💬 Request feedback survey
                    """)
                
                with col_b:
                    st.markdown("""
                    #### Follow-up Strategy (Week 3-4)
                    - 🚀 VIP early access to products
                    - 📦 Free shipping for 3 orders
                    - 🎯 Personalized recommendations
                    - 📊 Weekly engagement tracking
                    - 🔄 Re-engagement automation
                    """)
            else:
                st.markdown(f"""
                    <div class='prediction-box risk-low'>
                        ✅ LOW CHURN RISK - ENGAGED CUSTOMER
                        <div style='font-size: 15px; margin-top: 10px; font-weight: 500;'>
                        Retention Probability: {safe_prob:.1f}% | Customer: {customer_id}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("## 🚀 Growth & Expansion Strategy")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("""
                    #### Engagement & Loyalty
                    - 💎 Exclusive loyalty program
                    - 🎂 Birthday celebrations
                    - 🌟 Gamification & rewards
                    - 📱 App notifications
                    - ⭐ Encourage reviews
                    """)
                
                with col_b:
                    st.markdown("""
                    #### Revenue Expansion
                    - 🛍️ Smart cross-sell offers
                    - 📈 Bundle deals & discounts
                    - 👥 Referral bonuses
                    - 💳 Subscription enrollment
                    - 🎯 Premium tier upgrades
                    """)
        
        with col2:
            # Red gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "<b>Churn Risk Score</b>", 'font': {'size': 20, 'color': '#991b1b'}},
                number={'suffix': "%", 'font': {'size': 46, 'color': '#991b1b', 'family': 'Arial Black'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#991b1b"},
                    'bar': {'color': "#dc2626", 'thickness': 0.75},
                    'bgcolor': "white",
                    'borderwidth': 3,
                    'bordercolor': "#fca5a5",
                    'steps': [
                        {'range': [0, 30], 'color': '#fee2e2'},
                        {'range': [30, 50], 'color': '#fecaca'},
                        {'range': [50, 70], 'color': '#fca5a5'},
                        {'range': [70, 100], 'color': '#f87171'}
                    ],
                    'threshold': {
                        'line': {'color': "#991b1b", 'width': 4},
                        'thickness': 0.8,
                        'value': 50
                    }
                }
            ))
            
            fig_gauge.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=70, b=20),
                paper_bgcolor="rgba(255,255,255,0.95)"
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Risk badge
            risk_level = "🔴 Critical" if churn_prob > 70 else "🟡 Moderate" if churn_prob > 40 else "🟢 Low"
            risk_color = "#fee2e2" if churn_prob > 70 else "#fef3c7" if churn_prob > 40 else "#dcfce7"
            st.markdown(f"""
                <div style='text-align: center; padding: 20px; background: {risk_color}; 
                border-radius: 12px; margin-top: 15px; border: 2px solid #fca5a5;'>
                    <h3 style='margin: 0; color: #991b1b;'>Risk Level</h3>
                    <p style='font-size: 22px; font-weight: 700; margin: 8px 0; color: #7f1d1d;'>{risk_level}</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # --- BENCHMARKING ---
        if rfm_data is not None:
            st.markdown("## 📊 Customer Benchmarking Analysis")
            
            r_pct = (rfm_data['Recency'] >= recency).mean() * 100
            f_pct = (rfm_data['Frequency'] <= frequency).mean() * 100
            m_pct = (rfm_data['Monetary'] <= monetary).mean() * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📅 Recency Percentile", f"{r_pct:.0f}th",
                          "Above Average" if r_pct > 50 else "Below Average")
            
            with col2:
                st.metric("🔄 Frequency Percentile", f"{f_pct:.0f}th",
                          "Above Average" if f_pct > 50 else "Below Average")
            
            with col3:
                st.metric("💰 Monetary Percentile", f"{m_pct:.0f}th",
                          "Above Average" if m_pct > 50 else "Below Average")
            
            st.markdown("")
            
            col1, col2, col3 = st.columns(3)
            
            # Red histograms
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=rfm_data['Recency'],
                    nbinsx=30,
                    marker_color='#fca5a5',
                    showlegend=False
                ))
                fig.add_vline(x=recency, line_width=3, line_color="#991b1b", 
                              annotation_text="Current", annotation_position="top")
                fig.update_layout(
                    title="<b>Recency Distribution</b>",
                    height=260,
                    margin=dict(l=40, r=20, t=50, b=40),
                    paper_bgcolor="rgba(255,255,255,0.95)",
                    plot_bgcolor="white",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#fee2e2'),
                    font=dict(color='#991b1b')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=rfm_data['Frequency'],
                    nbinsx=20,
                    marker_color='#fca5a5',
                    showlegend=False
                ))
                fig.add_vline(x=frequency, line_width=3, line_color="#991b1b",
                              annotation_text="Current", annotation_position="top")
                fig.update_layout(
                    title="<b>Frequency Distribution</b>",
                    height=260,
                    margin=dict(l=40, r=20, t=50, b=40),
                    paper_bgcolor="rgba(255,255,255,0.95)",
                    plot_bgcolor="white",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#fee2e2'),
                    font=dict(color='#991b1b')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=rfm_data['Monetary'],
                    nbinsx=30,
                    marker_color='#fca5a5',
                    showlegend=False
                ))
                fig.add_vline(x=monetary, line_width=3, line_color="#991b1b",
                              annotation_text="Current", annotation_position="top")
                fig.update_layout(
                    title="<b>Monetary Distribution</b>",
                    height=260,
                    margin=dict(l=40, r=20, t=50, b=40),
                    paper_bgcolor="rgba(255,255,255,0.95)",
                    plot_bgcolor="white",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#fee2e2'),
                    font=dict(color='#991b1b')
                )
                st.plotly_chart(fig, use_container_width=True)

# --- SEGMENT ANALYSIS ---
st.markdown("---")
st.markdown("## 🔍 Customer Segment Overview")

if rfm_data is not None:
    def classify_segment(row):
        r, f, m = row['Recency'], row['Frequency'], row['Monetary']
        if r <= 30 and f >= 7 and m >= 1000:
            return "Champions"
        elif r <= 60 and f >= 4 and m >= 500:
            return "Loyal Customers"
        elif r <= 90 and f >= 2 and m >= 300:
            return "Potential Loyalists"
        elif r > 180 and f >= 4:
            return "At Risk"
        elif r > 180:
            return "Lost Customers"
        else:
            return "New Customers"
    
    rfm_data['Segment'] = rfm_data.apply(classify_segment, axis=1)
    
    segment_stats = rfm_data.groupby('Segment', as_index=False).agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'customer_unique_id': 'count'
    }).round(0)
    
    segment_stats.columns = ['Segment', 'Avg Recency (days)', 'Avg Frequency', 'Avg Monetary ($)', 'Total Customers']
    
    segment_stats = segment_stats.sort_values('Total Customers', ascending=False)
    
    segment_stats = segment_stats.dropna()
    segment_stats = segment_stats[segment_stats['Total Customers'] > 0]
    
    segment_stats.index = range(1, len(segment_stats) + 1)
    segment_stats.index.name = 'No.'
    
    col1, col2 = st.columns([1.5, 1], gap="large")
    
    with col1:
        st.markdown("### 📋 Segment Performance Metrics")
        
        st.dataframe(
            segment_stats.style.format({
                'Avg Recency (days)': '{:.0f}',
                'Avg Frequency': '{:.0f}',
                'Avg Monetary ($)': '${:.0f}',
                'Total Customers': '{:,.0f}'
            }),
            use_container_width=True,
        )
    
    with col2:
        st.markdown("### 🥧 Distribution by Segment")
        
        segment_counts = rfm_data['Segment'].value_counts().sort_values(ascending=False)
        
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            hole=0.45,
            color_discrete_sequence=['#991b1b', '#b91c1c', '#dc2626', '#ef4444', '#f87171', '#fca5a5']
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=12,
            textfont_color='white',
            marker=dict(line=dict(color='white', width=2))
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=20, b=20),
            showlegend=False,
            paper_bgcolor="rgba(255,255,255,0.95)"
        )
        
        st.plotly_chart(fig, use_container_width=True)


st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 25px; background: rgba(255,255,255,0.95); 
    border-radius: 12px; border: 2px solid #fca5a5;'>
        <h3 style='color: #991b1b; margin-bottom: 10px;'>🎯 Customer Churn Intelligence Platform</h3>
        <p style='color: #6b7280; font-size: 14px; margin: 5px 0;'>
            Powered by Machine Learning • Random Forest Classifier • Real-time Predictions
        </p>
        <p style='color: #9ca3af; font-size: 12px; margin-top: 10px;'>
            Built with Streamlit, Python, Scikit-learn & Plotly
        </p>
    </div>
""", unsafe_allow_html=True)
