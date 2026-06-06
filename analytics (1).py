import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.header("📊 E-Commerce Customer Analytics Dashboard")
st.markdown("Visualisasi data untuk memahami pola dan faktor-faktor yang memengaruhi churn pelanggan.")

df = st.session_state.get('df')
if df is None:
    st.error("Dataset tidak ditemukan. Pastikan file EComm.csv ada di direktori utama.")
    st.stop()

df_clean = df.dropna(subset=['Tenure', 'SatisfactionScore', 'CashbackAmount'])

tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔍 Churn Analysis", "📊 Feature Distribution"])

with tab1:
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        churn_rate = df['Churn'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with col3:
        st.metric("Avg Satisfaction", f"{df_clean['SatisfactionScore'].mean():.2f}/5")

    st.markdown("---")
    st.subheader("Sample Data")
    st.dataframe(df.head(10), width='stretch')

with tab2:
    st.subheader("Churn Rate by Category")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_by_device = df.groupby('PreferredLoginDevice')['Churn'].mean().sort_values(ascending=False)
        ax.bar(churn_by_device.index, churn_by_device.values, color=['#3B82F6', '#10B981'])
        ax.set_title('Churn by Login Device')
        ax.set_ylabel('Churn Rate')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_by_payment = df.groupby('PreferredPaymentMode')['Churn'].mean().sort_values(ascending=False)
        ax.barh(churn_by_payment.index, churn_by_payment.values, color='#F59E0B')
        ax.set_title('Churn by Payment Mode')
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_by_marital = df.groupby('MaritalStatus')['Churn'].mean().sort_values(ascending=False)
        ax.bar(churn_by_marital.index, churn_by_marital.values, color=['#8B5CF6', '#EC4899', '#6366F1'])
        ax.set_title('Churn by Marital Status')
        ax.set_ylabel('Churn Rate')
        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_by_category = df.groupby('PreferedOrderCat')['Churn'].mean().sort_values(ascending=False)
        ax.barh(churn_by_category.index, churn_by_category.values, color='#06B6D4')
        ax.set_title('Churn by Order Category')
        st.pyplot(fig)

with tab3:
    st.subheader("Feature Distribution Analysis")

    numeric_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'CashbackAmount', 'SatisfactionScore']

    selected_feature = st.selectbox("Select Feature for Distribution", numeric_cols)

    fig, ax = plt.subplots(figsize=(10, 4))
    churn_data = df[df['Churn']==1][selected_feature].dropna()
    retain_data = df[df['Churn']==0][selected_feature].dropna()

    ax.hist(retain_data, bins=30, alpha=0.7, label='Retained', color='#10B981', density=True)
    ax.hist(churn_data, bins=30, alpha=0.7, label='Churned', color='#EF4444', density=True)
    ax.set_xlabel(selected_feature)
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of {selected_feature} by Churn Status')
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Correlation Heatmap")
    numeric_df = df[['Churn', 'Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'SatisfactionScore',
                     'CashbackAmount', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder']].dropna()

    corr_cols = ['Churn', 'Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'SatisfactionScore',
                 'CashbackAmount', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder']
    corr_matrix = numeric_df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax, fmt='.2f')
    ax.set_title('Correlation Matrix: Churn vs Numerical Features')
    st.pyplot(fig)