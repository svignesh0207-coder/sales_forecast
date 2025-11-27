# app.py
import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Sales Forecast Pro",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

st.title("Sales Forecast Pro")
st.markdown("### Monthly Sales Forecasting using Facebook Prophet")

# Load model with fallback
@st.cache_resource
def load_model():
    try:
        return joblib.load("m.pkl")  # your model file
    except:
        st.error("Model not found! Using demo mode.")
        from prophet import Prophet
        dates = pd.date_range("2020-01-31", periods=48, freq='M')
        y = 50000 + 15000 * (1 + 0.5 * pd.Series(range(48))/48) + 8000 * pd.Series([i%12 for i in range(48)]).map(lambda x: (x<3 or x>9))
        df_demo = pd.DataFrame({'ds': dates, 'y': y + pd.Series(np.random.normal(0, 3000, 48))})
        m = Prophet(yearly_seasonality=True, seasonality_mode='multiplicative')
        m.fit(df_demo)
        return m

model = load_model()

# Sidebar
st.sidebar.header("Forecast Settings")
months = st.sidebar.slider("Forecast Horizon (Months)", 1, 36, 12)

if st.sidebar.button("Generate Forecast", type="primary"):
    with st.spinner("Forecasting..."):
        future = model.make_future_dataframe(periods=months, freq='ME')
        forecast = model.predict(future)

        # Split
        hist = model.history
        fc = forecast[forecast['ds'] > hist['ds'].max()]

        # Plot
        fig = go.Figure()

        # Historical
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'],
                                 mode='lines+markers', name='Historical',
                                 line=dict(color='#636EFA')))

        # Forecast
        fig.add_trace(go.Scatter(x=fc['ds'], y=fc['yhat'],
                                 mode='lines', name='Forecast',
                                 line=dict(color='#EF553B', width=3)))

        # Confidence interval
        fig.add_trace(go.Scatter(x=fc['ds'].tolist() + fc['ds'][::-1].tolist(),
                                 y=fc['yhat_upper'].tolist() + fc['yhat_lower'][::-1].tolist(),
                                 fill='toself', fillcolor='rgba(239,85,59,0.2)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 name='95% CI', showlegend=True))

        fig.update_layout(
            title="Monthly Sales Forecast",
            xaxis_title="Date",
            yaxis_title="Sales Quantity",
            hovermode="x unified",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        # Next month forecast
        next_month = fc.iloc[0]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Next Month Forecast", f"{int(next_month['yhat']):,}")
        with col2:
            st.metric("95% Confidence Range",
                      f"{int(next_month['yhat_lower']):,} – {int(next_month['yhat_upper']):,}")

        # Download forecast
        csv = fc[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        csv['ds'] = csv['ds'].dt.strftime('%Y-%m')
        csv = csv.round(0)
        st.download_button("Download Forecast CSV", csv.to_csv(index=False), "forecast.csv")

st.success("Model loaded successfully! Adjust settings and click **Generate Forecast**")
st.caption("Built with Streamlit • Model: Facebook Prophet")