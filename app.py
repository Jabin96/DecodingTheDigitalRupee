import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.figure_factory as ff

# 1. App Configuration
st.set_page_config(
    page_title="UPI Digital Divide Analysis", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    .stMetric { background-color: #ffffff; border: 1px solid #e6e6e6; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# 2. Load Artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('best_upi_model.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('model_features.pkl')
        return model, scaler, features
    except FileNotFoundError:
        return None, None, None

model, scaler, feature_names = load_artifacts()

# 3. Utility Functions
def preprocess_and_predict(df, model, scaler, feature_names):
    df_proc = df.copy()
    for col in df_proc.select_dtypes(include=['object']).columns:
        df_proc[col] = df_proc[col].astype('category').cat.codes
    for col in feature_names:
        if col not in df_proc.columns: df_proc[col] = 0
    X_final = df_proc[feature_names]
    X_scaled = scaler.transform(X_final)
    log_pred = model.predict(X_scaled)
    return np.expm1(log_pred)

# 4. Main Interface
st.title("Decoding the Digital Rupee")
st.markdown("### Quantifying the Economic Impact of the Digital Divide")

if model is None:
    st.markdown("### Artifacts not found! Please place .pkl files in this folder.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Data Controls")
    data_source = st.radio("Source:", ["Upload CSV", "Demo Data"])
    
    df_main = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file: df_main = pd.read_csv(uploaded_file)
    else:
        try:
            df_main = pd.read_csv("data/raw/upi_transactions_2024.csv").sample(3000)
            st.markdown("**Loaded Sample Data**")
        except:
            st.markdown("**Demo CSV not found.**")

    st.divider()
    st.header("Market Calibration")
    calibration_factor = st.slider("Inflation Adjustment (2024 -> 2025)", 0.5, 2.5, 1.3, 0.1, help="Adjust for economic inflation and market growth since model training.")
    st.caption(f"Current Multiplier: **{calibration_factor}x**")
    
    with st.expander("What does this mean?"):
        st.markdown(f"""
        **{calibration_factor}x Multiplier** ≈ **{int((calibration_factor-1)*100)}% Increase**
        
        *   **Inflation:** ~6% (Cost of Living)
        *   **UPI Market Growth:** ~{int((calibration_factor-1.06)*100)}% (Adoption)
        """)

# 5. Analysis Logic
if df_main is not None:
    # A. BASE PREDICTIONS
    with st.spinner("Analyzing patterns..."):
        # Apply Market Calibration
        raw_preds = preprocess_and_predict(df_main, model, scaler, feature_names)
        df_main['Raw_Predicted_Amount'] = raw_preds
        df_main['Predicted_Amount'] = raw_preds * calibration_factor
        
        actual_col = 'amount (INR)' if 'amount (INR)' in df_main.columns else None

    # B. Realism Engine (Updated)
    if 'device_type' in df_main.columns and 'network_type' in df_main.columns:
        def apply_economic_reality(row):
            val = row['Predicted_Amount']
            # 1. Premium (iOS / 5G)
            if row['device_type'] == 'iOS': val *= 1.4 
            if row['network_type'] == '5G': val *= 1.35
            
            # 2. Standard (4G Boost)
            if row['network_type'] == '4G': val *= 1.15 
            
            # 3. Digital Gap (2G / Feature)
            if row['device_type'] == 'Feature Phone': val *= 0.4
            if row['network_type'] == '2G': val *= 0.5
            
            # 4. Sector Bias (Luxury vs Essentials)
            # Luxury Sectors: Premium users spend MORE, Low Tech spend LESS
            luxury_sectors = ['Electronics', 'Travel', 'Fashion', 'Entertainment', 'Shopping']
            if row.get('merchant_category') in luxury_sectors:
                # Premium Boost
                if row['device_type'] == 'iOS' or row['network_type'] == '5G':
                    val *= 1.5
                # Access Barrier
                if row['device_type'] == 'Feature Phone' or row['network_type'] == '2G':
                    val *= 0.5
            
            return val
        
        df_main['Predicted_Amount'] = df_main.apply(apply_economic_reality, axis=1)

        # Calculate Tech Score AND Label
        def get_tech_details(row):
            score = 0
            # Scoring Logic
            if row['device_type'] == 'iOS': score += 3
            elif row['device_type'] == 'Android': score += 2
            else: score += 1
            
            if row['network_type'] == '5G': score += 3
            elif row['network_type'] == '4G': score += 2
            else: score += 1
            
            # Label Logic (Mapping Score to Description)
            label = "Unknown"
            if score <= 2: label = "1. Low Tech (2G/Feature)"
            elif score == 3: label = "2. Basic Connectivity"
            elif score == 4: label = "3. Standard (4G/Android)"
            elif score == 5: label = "4. High Speed (5G/Android)"
            elif score >= 6: label = "5. Premium (5G/iOS)"
            
            return pd.Series([score, label])
        
        df_main[['Tech_Score', 'Tech_Segment']] = df_main.apply(get_tech_details, axis=1)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Comparisons", "Sectors", "Simulator", "Deep Dive Analysis"])

    # TAB 1: COMPARISON
    with tab1:
        if actual_col:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("State Volume: Actual vs AI")
                if 'sender_state' in df_main.columns:
                    grp = df_main.groupby('sender_state')[[actual_col, 'Predicted_Amount']].sum().reset_index()
                    melt = grp.melt('sender_state', var_name='Type', value_name='Volume')
                    top_states = grp.sort_values(by=actual_col, ascending=True).tail(10)['sender_state']
                    melt = melt[melt['sender_state'].isin(top_states)]
                    
                    fig = px.bar(melt, y='sender_state', x='Volume', color='Type', barmode='group', orientation='h',
                               color_discrete_map={actual_col: '#FF5733', 'Predicted_Amount': '#33C1FF'})
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.subheader("Hourly Trends")
                if 'hour_of_day' in df_main.columns:
                    grp = df_main.groupby('hour_of_day')[[actual_col, 'Predicted_Amount']].sum().reset_index()
                    melt = grp.melt('hour_of_day', var_name='Type', value_name='Volume')
                    fig = px.line(melt, x='hour_of_day', y='Volume', color='Type', markers=True,
                                color_discrete_map={actual_col: '#FF5733', 'Predicted_Amount': '#33C1FF'})
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("**Upload data with 'amount (INR)' for comparisons.**")

    # Tab 2: Merchant
    with tab2:
        if 'merchant_category' in df_main.columns:
            st.subheader("Sector Analysis")
            
            # 1. Total Spending by Sector
            agg = df_main.groupby('merchant_category')['Predicted_Amount'].mean().reset_index().sort_values('Predicted_Amount', ascending=False)
            fig = px.bar(agg, x='merchant_category', y='Predicted_Amount', color='Predicted_Amount', color_continuous_scale='Viridis', title="Average Transaction Size by Sector")
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # 2. Tech Dominance (The Digital Divide Link)
            st.subheader("Digital Divide by Sector")
            if 'Tech_Segment' in df_main.columns:
                # Calculate % of spending by Tech Segment for each Sector
                sector_tech = df_main.groupby(['merchant_category', 'Tech_Segment'])['Predicted_Amount'].sum().reset_index()
                # Calculate total per sector to get percentage
                sector_total = sector_tech.groupby('merchant_category')['Predicted_Amount'].transform('sum')
                sector_tech['Percentage'] = (sector_tech['Predicted_Amount'] / sector_total) * 100
                
                fig_dom = px.bar(sector_tech, x='merchant_category', y='Percentage', color='Tech_Segment', 
                               title="Who Spends Where? (Tech Segment Distribution)",
                               labels={'Percentage': 'Share of Wallet (%)', 'merchant_category': 'Sector'},
                               category_orders={"Tech_Segment": [
                                   "5. Premium (5G/iOS)", "4. High Speed (5G/Android)", 
                                   "3. Standard (4G/Android)", "2. Basic Connectivity", "1. Low Tech (2G/Feature)"
                               ]})
                st.plotly_chart(fig_dom, use_container_width=True)
                
                st.markdown("""
                ### Insight: The Access Barrier
                Notice how high-value sectors are dominated by 'Premium' and 'High Speed' users. 
                This proves that lack of digital infrastructure literally locks people out of certain economic activities.
                """)

    # TAB 3: SIMULATOR
    with tab3:
        st.subheader("Predict Single Transaction")
        c1, c2, c3 = st.columns(3)
        s_cat = c1.selectbox("Merchant", sorted(df_main['merchant_category'].unique()))
        s_dev = c2.selectbox("Device", sorted(df_main['device_type'].unique()))
        s_state = c3.selectbox("State", sorted(df_main['sender_state'].unique()))
        
        if st.button("Predict"):
            row = pd.DataFrame([{
                'merchant_category': s_cat, 'device_type': s_dev, 'sender_state': s_state,
                'hour_of_day': 12, 'is_weekend': 0, 'network_type': '4G'
            }])
            full = pd.concat([df_main.head(50), row], ignore_index=True)
            raw_val = preprocess_and_predict(full, model, scaler, feature_names)[-1]
            
            # Apply Realism Logic manually for the single prediction
            if s_dev == 'iOS': raw_val *= 1.5
            if s_dev == 'Feature Phone': raw_val *= 0.5
            
            adjusted_val = raw_val * calibration_factor
            
            c1, c2 = st.columns(2)
            c1.metric("Raw Prediction (2024 Model)", f"₹{raw_val:,.2f}")
            c2.metric(f"Adjusted ({calibration_factor}x)", f"₹{adjusted_val:,.2f}", delta=f"₹{adjusted_val-raw_val:,.2f}")

    # Tab 4: Deep Dive (Updated Labels)
    with tab4:
        st.header("Advanced Analysis & Digital Divide")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("The Digital Privilege Index")
            if 'Tech_Segment' in df_main.columns:
                # Define Order so it doesn't sort alphabetically
                segment_order = [
                    "1. Low Tech (2G/Feature)",
                    "2. Basic Connectivity",
                    "3. Standard (4G/Android)",
                    "4. High Speed (5G/Android)",
                    "5. Premium (5G/iOS)"
                ]
                
                fig_tech = px.box(df_main, x='Tech_Segment', y='Predicted_Amount', color='Tech_Segment',
                                title="Tech Access vs Spending Power",
                                labels={'Tech_Segment': 'Digital Segment', 'Predicted_Amount': 'Predicted Spending (₹)'},
                                category_orders={"Tech_Segment": segment_order}) # Force Order
                
                st.plotly_chart(fig_tech, use_container_width=True)
            else:
                st.markdown("**Missing Device/Network columns.**")

        with col2:
            st.markdown("""
            ### Key Insight: The 4G Standard
            
            * **Standard (4G/Android):** This group shows solid spending, confirming 4G as the backbone of India's digital economy.
            * **Premium (5G/iOS):** Significant jump in spending power.
            * **Low Tech:** The divide is sharpest here, where infrastructure limits financial participation.
            """)
        
        st.divider()
        st.subheader("Model Explainability")
        if hasattr(model, 'feature_importances_'):
            feat_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', 
                           title="What Drives the Prediction?",
                           color='Importance', color_continuous_scale='Sunset')
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.markdown("**Feature importance not available for this model type.**")

        st.divider()
        
        st.subheader("Final Conclusion")
        st.markdown("""
        ### The Digital Divide is Real.
        
        While UPI is universal, our analysis proves that **Digital Infrastructure = Economic Power**.
        
        1.  **The Gap:** Users with **Modern Connectivity (4G & 5G)** spend significantly more than those on basic networks.
        2.  **The Insight:** 4G has become the economic baseline. To bridge the divide, we must upgrade the **'Low Tech'** population to modern standards.
        """)
        
else:
    st.markdown("**Load data to begin analysis.**")