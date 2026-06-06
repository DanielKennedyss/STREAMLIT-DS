import streamlit as st
import pandas as pd
import numpy as np

st.header("🔮 Live Churn Engine - Individual Prediction")
st.markdown("Masukkan data pelanggan di sidebar lalu klik tombol di bawah untuk memprediksi risiko churn.")

# Ambil dari session_state yang di-passing dari app.py
model = st.session_state.get('model') or None
scaler = st.session_state.get('scaler') or None
inputs = st.session_state.get('input_values', {})

# Fallback jika model/scaler gagal diload
if model is None or scaler is None:
    st.warning("⚠️ Model artifacts (`best_churn_model.pkl` atau `robust_scaler.pkl`) tidak ditemukan. Menampilkan simulasi...")
    if st.button("Run Simulation Prediction", type="primary"):
        simulated_churn = np.random.choice([0, 1], p=[0.7, 0.3])
        if simulated_churn == 1:
            st.error("🚨 HIGH RISK: Pelanggan ini diprediksi memiliki kecenderungan kuat untuk CHURN!")
        else:
            st.success("💚 LOYAL: Pelanggan ini diprediksi akan tetap bertahan (RETAINED).")
else:
    if st.button("Execute Real-Time Prediction", type="primary"):
        
        # 1. Ambil data input dari dictionary
        input_dict = {
            'Tenure': inputs.get('tenure', 12),
            'WarehouseToHome': inputs.get('warehouse_to_home', 15),
            'HourSpendOnApp': inputs.get('hour_spend_on_app', 3),
            'NumberOfDeviceRegistered': inputs.get('device_registered', 2),
            'SatisfactionScore': inputs.get('satisfaction_score', 3),
            'NumberOfAddress': inputs.get('number_of_address', 3),
            'Complain': inputs.get('complain_val', 0),
            'OrderAmountHikeFromlastYear': inputs.get('order_hike', 10),
            'CouponUsed': inputs.get('coupon_used', 2),
            'OrderCount': inputs.get('order_count', 4),
            'DaySinceLastOrder': inputs.get('day_since_last_order', 5),
            'CashbackAmount': inputs.get('cashback_amount', 150),
            'PreferredLoginDevice': inputs.get('login_device', 'Mobile Phone'),
            'PreferredPaymentMode': inputs.get('payment_mode', 'Debit Card'),
            'Gender': inputs.get('gender', 'Female'),
            'PreferedOrderCat': inputs.get('order_cat', 'Laptop & Accessory'),
            'MaritalStatus': inputs.get('marital_status', 'Single')
        }

        # 2. Ubah dictionary menjadi DataFrame 1 baris
        df_input = pd.DataFrame([input_dict])

        # 3. Lakukan One-Hot Encoding pada kolom kategori
        categorical_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']
        df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True, dtype=int)

        # 4. Bersihkan nama kolom dari karakter spesial dan spasi (agar sesuai format XGBoost)
        df_encoded.columns = df_encoded.columns.str.replace(r'[^\w\s]', '', regex=True)
        df_encoded.columns = df_encoded.columns.str.replace(' ', '_')

        try:
            # ==========================================
            # PROSES PENYESUAIAN KOLOM & SCALING FINAL
            # ==========================================
            
            # 5. Ambil nama kolom numerik yang dikenali oleh scaler (11 kolom)
            num_cols = scaler.feature_names_in_
            
            # Pastikan semua kolom numerik ada di df_encoded sebelum di-scale
            for col in num_cols:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
                    
            # 6. Lakukan scaling HANYA pada 11 kolom numerik tersebut
            df_encoded[num_cols] = scaler.transform(df_encoded[num_cols])

            # 7. Ambil daftar pasti 29 kolom yang dibutuhkan oleh model XGBoost
            model_cols = model.feature_names_in_

            # 8. Sesuaikan df_encoded agar pas menjadi 29 kolom. 
            # Kolom kategori yang kosong akan otomatis dibuatkan dan diisi angka 0.
            df_final = df_encoded.reindex(columns=model_cols, fill_value=0)

            # ==========================================
            # EKSEKUSI PREDIKSI
            # ==========================================
            
            prediction = model.predict(df_final)[0]
            probability = model.predict_proba(df_final)[0][1]

            # 9. Tampilkan Hasil di Streamlit
            st.markdown("---")
            st.subheader("📋 Prediction Result")
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error(f"🚨 CHURN RISK: {probability*100:.1f}%")
                else:
                    st.success(f"💚 LOYAL: {(1-probability)*100:.1f}%")
                    
            with col2:
                if prediction == 1:
                    st.info("**Rekomendasi:** Lakukan intervensi retensi segera! Tawarkan diskon atau *follow-up* terkait komplain pelanggan.")
                else:
                    st.info("**Rekomendasi:** Pertahankan *engagement* yang baik. Pelanggan ini menunjukkan loyalitas yang stabil.")

        except Exception as e:
            st.error(f"⚠️ Terjadi kesalahan pada pemrosesan model: {e}")
            st.info("Pastikan nama kolom dataset saat melatih scaler dan model sudah sama persis.")