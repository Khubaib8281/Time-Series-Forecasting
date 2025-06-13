import streamlit as st
import pandas as pd
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

# Set wide layout
st.set_page_config(page_title="Favorita Sales Forecasting", layout="wide")

# Custom CSS for sleek UI


import streamlit as st

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&family=Roboto:wght@300;400;500&display=swap');

    html, body, .stApp {
        font-family: 'Roboto', sans-serif;
        color: #212121;
        background-color: #F5F7FA;
        line-height: 1.65;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif;
        color: #FFFFFF;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    div{
        color: #708090
    }

    p, li {
        color: #708090;
    }

    .big-font {
        font-family: 'Montserrat', sans-serif;
        font-size: 3.2em !important;
        font-weight: 700;
        color: #FFFFFF;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        margin-bottom: 35px;
        text-align: center;
        letter-spacing: -0.02em;
    }

    .metric-box {
        background: linear-gradient(135deg, #E0F2F7, #B2EBF2);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px rgba(0, 70, 100, 0.2);
        transition: all 0.4s ease-in-out;
        border: 1px solid #81D4FA;
        color: #0A304D;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        text-align: center;
        font-size: 1.4em;
        line-height: 1.3;
    }

    .metric-box:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 35px rgba(0, 70, 100, 0.35);
        background: linear-gradient(135deg, #B2EBF2, #81D4FA);
    }

    .section {
        background-color: #FFFFFF;
        padding: 40px;
        border-radius: 25px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
        margin-bottom: 60px;
        border: 1px solid #E0E0E0;
        color: #212121;
    }

    .footer {
        text-align: center;
        color: #616161;
        font-size: 0.95em;
        padding-top: 40px;
        border-top: 1px solid #EEEEEE;
        font-family: 'Roboto', sans-serif;
    }

    /* Selectbox base style */
    div[data-baseweb="select"] {
        background-color: #000000 !important;
        color: #000000 !important;
        border-radius: 10px;
        border: 1px solid #81D4FA !important;
    }

    div[data-baseweb="select"] * {
        color: #FFFFFF !important;
    }

    ul[role="listbox"] {
        background-color: #F0FAFF !important;
        color: #FFFFFF !important;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    ul[role="listbox"] li {
        padding: 10px;
        font-family: 'Roboto', sans-serif;
        transition: background 0.2s ease;
        color: #212121;
    }

    ul[role="listbox"] li:hover {
        background-color: #212121 !important; /* Black/dark hover */
        color: #FFFFFF !important; /* White text on dark background */
        cursor: pointer;
    }

    /* Button styles – white text on dark background */
    button[kind="primary"] {
        background-color: #212121 !important;
        color: #FFFFFF !important;
        border-radius: 10px;
        border: none;
        font-weight: 600;
    }

    button[kind="primary"]:hover {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        transform: scale(1.03);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #D0F0C0, #A0E7E5);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0, 150, 100, 0.2);
        text-align: center;
        font-family: 'Montserrat', sans-serif;
        color: #004D40;
        font-size: 1.6em;
        font-weight: 700;
        margin-top: 25px;
        transition: 0.3s ease-in-out;
        border: 2px solid #B2DFDB;
    }
    .prediction-box:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 30px rgba(0, 150, 100, 0.3);
    }
    </style>
""", unsafe_allow_html=True)


# Load models
rf_pipeline = joblib.load('RF_model.pkl')
xg_pipeline = joblib.load('XGBR_model.pkl')
sarimax_model = SARIMAXResultsWrapper.load('SARIMAX_model.pkl')

# Header
st.markdown('<h1 style="color:#ff4b4b;">🛍️ Favorita Sales Forecasting Dashboard</h1>', unsafe_allow_html=True)
st.markdown("Predict sales using **SARIMAX**, **Random Forest**, or **XGBoost** trained on Ecuador's grocery data.")

st.markdown("---")

# Model Selector
model_choice = st.selectbox("📌 Select Model Type", ["SARIMAX", "Random Forest", "XGBoost"])

# ──────────────────────────────────────────
# ML Models Section
# ──────────────────────────────────────────
if model_choice in ["Random Forest", "XGBoost"]:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader(f"🧮 Input Features for {model_choice}")

    col1, col2, col3 = st.columns(3)
    with col1:
        family = st.selectbox("Family", ['AUTOMOTIVE', 'BEAUTY'])
        store_nbr = st.number_input("Store Number", value=54)
        onpromotion = st.number_input("On Promotion", value=12)

    with col2:
        type_holiday = st.selectbox("Holiday Type", ['None', 'Holiday', 'Event', 'Additional'])
        dcoilwtico = st.number_input("Oil Price (WTI)", value=65.32)
        transactions = st.number_input("Transactions", value=1789.0)

    with col3:
        cluster = st.number_input("Cluster", value=7)
        sales_lag_1 = st.number_input("Sales Lag (1 day)", value=230.4)
        sales_rolling_7 = st.number_input("7-Day Rolling Avg Sales", value=215.7)

    col4, col5, col6 = st.columns(3)
    with col4: day = st.number_input("Day", value=23)
    with col5: month = st.number_input("Month", value=7)
    with col6: year = st.number_input("Year", value=2017)

    input_df = pd.DataFrame({
        'family': [family],
        'type_holiday': [type_holiday],
        'store_nbr': [store_nbr],
        'onpromotion': [onpromotion],
        'dcoilwtico': [dcoilwtico],
        'transactions': [transactions],
        'cluster': [cluster],
        'sales_lag_1': [sales_lag_1],
        'sales_rolling_7': [sales_rolling_7],
        'day': [day],
        'month': [month],
        'year': [year]
    })

    if st.button("🎯 Predict Sales"):
        if model_choice == "Random Forest":
            prediction = rf_pipeline.predict(input_df)[0]
            r2_score = "0.68"
            st.markdown("#### 📈 Model Evaluation Metrics")
            colm1, colm2, colm3 = st.columns(3)
            with colm1:
                st.markdown('<div class="metric-box"><b>MAE:</b> 1.48</div>', unsafe_allow_html=True)
            with colm2:
                st.markdown('<div class="metric-box"><b>RMSE:</b> 2.14</div>', unsafe_allow_html=True)
            with colm3:
                st.markdown(f'<div class="metric-box"><b>R² Score:</b> {r2_score}</div>', unsafe_allow_html=True)
        else:
            prediction = xg_pipeline.predict(input_df)[0]
            r2_score = "0.65"
            st.markdown("#### 📈 Model Evaluation Metrics")
            colm1, colm2, colm3 = st.columns(3)
            with colm1:
                st.markdown('<div class="metric-box"><b>MAE:</b> 1.48</div>', unsafe_allow_html=True)
            with colm2:
                st.markdown('<div class="metric-box"><b>RMSE:</b> 2.14</div>', unsafe_allow_html=True)
            with colm3:
                st.markdown(f'<div class="metric-box"><b>R² Score:</b> {r2_score}</div>', unsafe_allow_html=True)

        st.markdown(f"""
            <div class='prediction-box'>
                📊 Predicted Sales: <span style='color:#00695C;'>{prediction:.2f} units</span>
            </div>
        """, unsafe_allow_html=True)


    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────
# SARIMAX Section
# ──────────────────────────────────────────
else:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📈 Forecast Using SARIMAX")
    st.markdown("Enter past `sales_rolling_7` values to forecast next **7 days**.")

    col1, col2 = st.columns([1, 2])
    with col1:
        date_input = st.date_input("🗓️ Forecast Start Date", value=pd.to_datetime('2017-08-16'))

    with col2:
        sales_input = st.text_input("📥 7 comma-separated `sales_rolling_7` values",
                                    "42.1, 43.2, 44.0, 45.3, 46.1, 47.5, 48.0")

    try:
        sales_vals = [float(v.strip()) for v in sales_input.split(",")]
        if len(sales_vals) != 7:
            st.warning("⚠️ Please enter exactly 7 values.")
        else:
            start_date = pd.to_datetime(date_input)
            future_index = pd.date_range(start=start_date, periods=7, freq='D')
            exog_df = pd.DataFrame({'sales_rolling_7': sales_vals}, index=future_index)

            forecast = sarimax_model.get_forecast(steps=7, exog=exog_df)
            prediction = forecast.predicted_mean
            prediction.index = future_index

            st.success("✅ Forecast Generated Successfully!")

            st.subheader("🔮 7-Day Forecasted Sales")
            st.dataframe(prediction.reset_index().rename(columns={'index': 'Date', 0: 'Forecasted Sales'}))

            st.markdown("#### 📊 SARIMAX Evaluation Metrics")
            m1, m2 = st.columns(2)
            with m1:
                st.markdown('<div class="metric-box"><b>MAE:</b> 2.27</div>', unsafe_allow_html=True)
            with m2:
                st.markdown('<div class="metric-box"><b>RMSE:</b> 2.76</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Created by Muhammad Khubaib Ahmad | Favorita Dataset | Streamlit App</div>', unsafe_allow_html=True)
