import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="AQI Prediction Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.title("🌍 Air Quality Index (AQI) Prediction Dashboard")

# API URL
API_URL = "http://localhost:8000"

# Color mapping for categories
CATEGORY_COLORS = {
    "Good": "#1f77b4",
    "Moderate": "#ffdd57",
    "Unhealthy for Sensitive Groups": "#ff7043",
    "Unhealthy": "#e53935",
    "Very Unhealthy": "#7e1f86",
    "Hazardous": "#4a148c"
}

# Initialize session state for predictions
if 'predictions_data' not in st.session_state:
    st.session_state.predictions_data = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# Sidebar with refresh button
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Make Prediction", width='stretch', key="predict_btn"):
        st.session_state.predictions_data = None
        st.session_state.last_update = None

# Main content
col1, col2, col3, col4 = st.columns(4)

# Container for status
status_placeholder = st.empty()

# Fetch predictions on button click
if st.session_state.predictions_data is None:
    with status_placeholder.container():
        with st.spinner("📊 Loading AQI predictions..."):
            try:
                response = requests.get(f"{API_URL}/api/predict", timeout=30)
                response.raise_for_status()
                st.session_state.predictions_data = response.json()
                st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to prediction API. Make sure the FastAPI server is running on http://localhost:8000")
                st.session_state.predictions_data = None
            except Exception as e:
                st.error(f"❌ Error fetching predictions: {str(e)}")
                st.session_state.predictions_data = None

# Display content if data is available
if st.session_state.predictions_data:
    data = st.session_state.predictions_data
    status_placeholder.success(f"✅ Last updated: {st.session_state.last_update}")

    # Today's AQI Section (Predicted)
    st.markdown("---")
    st.subheader("📍 Today's AQI (Predicted)")

    predictions = data.get('predictions', [])
    if predictions:
        today_pred = predictions[0]
        aqi_value = today_pred.get('aqi')
        category = today_pred.get('category', 'Unknown')
        color = today_pred.get('color', '#808080')

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.metric(
                label="Today's AQI",
                value=f"{aqi_value:.1f}" if aqi_value else "N/A",
                delta=category,
                delta_color="off"
            )

        with col2:
            st.info(f"**Category:** {category}")
            st.write(f"**Date:** {today_pred.get('date', 'N/A')}")

        with col3:
            # Visual indicator
            fig = go.Figure(data=[go.Indicator(
                mode="gauge+number",
                value=aqi_value if aqi_value else 0,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 500]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 50], 'color': "#1f77b4"},
                        {'range': [50, 100], 'color': "#ffdd57"},
                        {'range': [100, 150], 'color': "#ff7043"},
                        {'range': [150, 200], 'color': "#e53935"},
                        {'range': [200, 300], 'color': "#7e1f86"},
                        {'range': [300, 500], 'color': "#4a148c"}
                    ]
                }
            )])
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
            st.plotly_chart(fig, width='stretch')

    # Predictions Section (remaining days)
    st.markdown("---")
    st.subheader("📈 Next 4 Days Forecast")
    if predictions:
        pred_cols = st.columns(len(predictions))
        for idx, (col, pred) in enumerate(zip(pred_cols, predictions)):
            with col:
                day_label = ["Today", "Tomorrow", "Day 2", "Day 3"][idx] if idx < 4 else f"Day {idx}"
                category = pred.get('category', 'Unknown')
                aqi = pred.get('aqi', 0)

                st.metric(
                    label=f"{day_label}\n{pred.get('date', 'N/A')}",
                    value=f"{aqi:.1f}",
                    delta=category,
                    delta_color="off"
                )

                # Category badge
                color = CATEGORY_COLORS.get(category, '#808080')
                st.markdown(
                    f"<div style='background-color:{color};padding:10px;border-radius:5px;text-align:center;color:white;font-weight:bold;'>"
                    f"{category}</div>",
                    unsafe_allow_html=True
                )
    else:
        st.warning("No predictions available")

    # Last 24 Hours Data - Tabbed Variable Charts
    st.markdown("---")
    st.subheader("📊 Last 24 Hours - Variables")

    hourly_data = data.get('hourly_today', [])
    if hourly_data:
        df_hourly = pd.DataFrame(hourly_data)
        df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'])
        df_hourly = df_hourly.sort_values('datetime')

        # Get all variables except datetime and derived features
        all_vars = [col for col in df_hourly.columns if col not in ['datetime'] and
                   'rolling' not in col and 'lag' not in col and 'sin' not in col and
                   'cos' not in col and 'is_rush' not in col and 'is_work' not in col and
                   'precip' not in col and 'pm_ratio' not in col and 'aqi_change' not in col]

        if all_vars:
            tabs = st.tabs(all_vars)

            for tab, var in zip(tabs, all_vars):
                with tab:
                    if var in df_hourly.columns:
                        valid_data = df_hourly[['datetime', var]].dropna()

                        if len(valid_data) > 0:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=valid_data['datetime'],
                                y=valid_data[var],
                                mode='lines+markers',
                                name=var,
                                line=dict(color='#1f77b4', width=2),
                                marker=dict(size=6),
                                fill='tozeroy',
                                fillcolor='rgba(31, 119, 180, 0.2)'
                            ))

                            fig.update_layout(
                                title=f"Last 24 Hours - {var}",
                                xaxis_title="Time",
                                yaxis_title=var,
                                hovermode='x unified',
                                height=400,
                                template="plotly_dark",
                                showlegend=False
                            )
                            st.plotly_chart(fig, width='stretch')
                        else:
                            st.info(f"No data available for {var}")
        else:
            st.info("No variables available")
    else:
        st.info("No hourly data available")

    # Model Information
    st.markdown("---")
    st.subheader("🤖 Model Information")

    model_info = data.get('model_info', {})
    model_name = model_info.get('name', 'Unknown')
    r2 = model_info.get('r2', 0)
    rmse = model_info.get('rmse', 0)
    mae = model_info.get('mae', 0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Selected Model", model_name)
    with col2:
        st.metric("R² Score", f"{r2:.4f}" if r2 else "N/A")
    with col3:
        st.metric("RMSE", f"{rmse:.2f}" if rmse else "N/A")
    with col4:
        st.metric("MAE", f"{mae:.2f}" if mae else "N/A")

    # All Models Comparison
    st.markdown("---")
    st.subheader("📊 All Models Performance")

    all_models = data.get('all_models', [])
    if all_models:
        df_models = pd.DataFrame(all_models)
        df_models = df_models.rename(columns={'name': 'Model', 'r2': 'R²', 'rmse': 'RMSE', 'mae': 'MAE'})

        # Round values
        for col in ['R²', 'RMSE', 'MAE']:
            if col in df_models.columns:
                df_models[col] = df_models[col].round(4)

        st.dataframe(df_models, width='stretch', hide_index=True)

        # Visualization
        col1, col2 = st.columns(2)

        with col1:
            fig_r2 = px.bar(df_models, x='Model', y='R²', title='R² Score Comparison', color='R²',
                           color_continuous_scale='Viridis')
            st.plotly_chart(fig_r2, width='stretch')

        with col2:
            fig_rmse = px.bar(df_models, x='Model', y='RMSE', title='RMSE Comparison', color='RMSE',
                             color_continuous_scale='Reds')
            st.plotly_chart(fig_rmse, width='stretch')
    else:
        st.warning("No model metrics available")

    # SHAP Summary Plot
    st.markdown("---")
    st.subheader("🔍 Model Explainability - SHAP Summary")

    try:
        with st.spinner("Generating SHAP explanations..."):
            # Fetch raw data for SHAP
            shap_response = requests.get(f"{API_URL}/api/shap-data", timeout=30)
            if shap_response.status_code == 200:
                shap_data = shap_response.json()
                X_sample = pd.DataFrame(shap_data['X_sample'])
                shap_values = np.array(shap_data['shap_values'])
                feature_names = shap_data['feature_names']

                # Create SHAP summary plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                                 plot_type="dot", show=False, plot_size=(10, 6))
                st.pyplot(fig, width='stretch')
                plt.close()
            else:
                st.info("⚠️ SHAP data not available yet (model may need more data)")
    except Exception as e:
        st.info(f"⚠️ Could not generate SHAP plot: {str(e)[:100]}")

    # Footer
    st.markdown("---")
    st.caption(
        "🌱 AQI Prediction System | "
        "Data updated hourly | "
        "For more info visit your local environmental agency"
    )
