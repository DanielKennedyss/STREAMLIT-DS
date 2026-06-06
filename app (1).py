import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(
    page_title="E-Commerce Churn Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOAD TRAINED ARTIFACTS (Model & Scaler) & DATASET
# ==============================================================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('best_churn_model.pkl')
        scaler = joblib.load('robust_scaler.pkl')
        return model, scaler
    except:
        return None, None

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('EComm.csv')
        return df
    except:
        return None

model, scaler = load_artifacts()
df = load_dataset()
st.session_state.df = df
st.session_state.model = model
st.session_state.scaler = scaler

# ==============================================================================
# 3. SIDEBAR - INDIVIDUAL CUSTOMER INPUT FORM
# ==============================================================================
st.sidebar.header("👤 Customer Profile Input")
st.sidebar.markdown("Masukkan data perilaku pelanggan untuk memprediksi risiko *churn*.")

# --- Numerical Inputs ---
st.sidebar.subheader("Numerical Metrics")
tenure = st.sidebar.slider("Tenure (Months)", 0, 60, 12)
warehouse_to_home = st.sidebar.slider("Warehouse to Home Distance (km)", 1, 100, 15)
hour_spend_on_app = st.sidebar.slider("Hours Spent on App", 1, 8, 3)
device_registered = st.sidebar.slider("Devices Registered", 1, 6, 2)
satisfaction_score = st.sidebar.slider("Satisfaction Score", 1, 5, 3)
number_of_address = st.sidebar.slider("Number of Saved Addresses", 1, 20, 3)
complain = st.sidebar.selectbox("Has Active Complain?", ["No", "Yes"])
order_hike = st.sidebar.slider("Order Amount Hike From Last Year (%)", 0, 50, 10)
coupon_used = st.sidebar.number_input("Coupons Used This Month", min_value=0, max_value=50, value=2)
order_count = st.sidebar.number_input("Total Order Count", min_value=1, max_value=100, value=4)
day_since_last_order = st.sidebar.slider("Days Since Last Order", 0, 30, 5)
cashback_amount = st.sidebar.number_input("Average Cashback Amount ($)", min_value=0, max_value=500, value=150)

# --- Categorical Inputs ---
st.sidebar.subheader("Categorical Preferences")
login_device = st.sidebar.selectbox("Preferred Login Device", ["Mobile Phone", "Computer"])
payment_mode = st.sidebar.selectbox("Preferred Payment Mode", ["Debit Card", "Credit Card", "E Wallet", "COD", "UPI"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
order_cat = st.sidebar.selectbox("Preferred Order Category", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])

complain_val = 1 if complain == "Yes" else 0

# Store input values in session state
input_values = {
    'tenure': tenure, 'warehouse_to_home': warehouse_to_home,
    'hour_spend_on_app': hour_spend_on_app, 'device_registered': device_registered,
    'satisfaction_score': satisfaction_score, 'number_of_address': number_of_address,
    'complain_val': complain_val, 'order_hike': order_hike,
    'coupon_used': coupon_used, 'order_count': order_count,
    'day_since_last_order': day_since_last_order, 'cashback_amount': cashback_amount,
    'login_device': login_device, 'payment_mode': payment_mode,
    'gender': gender, 'order_cat': order_cat, 'marital_status': marital_status
}
st.session_state.input_values = input_values

# ==============================================================================
# 4. MAIN PAGE LAYOUT
# ==============================================================================
st.markdown("<div class='main-title'>🌊 E-Commerce Customer Churn Analytics Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Powered by Advanced ML Architecture (Team: Cry Me a River)</div>", unsafe_allow_html=True)

pg = st.navigation([
    st.Page("views/home.py", title="🏠 Home & Prediction", icon="🔮"),
    st.Page("views/analytics.py", title="📊 Data Analytics", icon="📈")
])
pg.run()