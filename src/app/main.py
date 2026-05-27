import pickle
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from src.utils.db_utils import fetch_features, load_latest_model

app = FastAPI(title="AQI Prediction API")

app = FastAPI(title="AQI Prediction API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_aqi_category(aqi_value):
    """Map AQI value to category and color."""
    if aqi_value <= 50:
        return "Good", "green"
    elif aqi_value <= 100:
        return "Moderate", "yellow"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "orange"
    elif aqi_value <= 200:
        return "Unhealthy", "red"
    elif aqi_value <= 300:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "maroon"


def forecast_96_hours(df, model, features, scaler):
    """
    Perform a 96-hour iterative forecast using the trained model.

    Parameters:
    df (pd.DataFrame): Historical dataframe containing 'datetime', 'us_aqi', and other features.
    model: Your trained model.
    features (list): The list of feature columns required for the model prediction.
    scaler: MinMaxScaler fitted on training data.
    """
    sim_df = df.copy()

    if not np.issubdtype(sim_df['datetime'].dtype, np.datetime64):
        sim_df['datetime'] = pd.to_datetime(sim_df['datetime'], utc=True).dt.tz_convert('Asia/Karachi').dt.tz_localize(None)
    sim_df = sim_df.sort_values('datetime').reset_index(drop=True)

    if len(sim_df) == 0:
        raise ValueError("Input dataframe is empty")

    target_var = 'us_aqi'
    rolling_mean_windows = [36]
    rolling_std_windows = [72]

    predictions = []

    for step in range(96):
        try:
            last_row = sim_df.iloc[-1].copy()
            next_time = last_row['datetime'] + pd.Timedelta(hours=1)

            new_row = last_row.copy()
            new_row['datetime'] = next_time
            new_row[target_var] = np.nan

            sim_df = pd.concat([sim_df, pd.DataFrame([new_row])], ignore_index=True)

            last_idx = len(sim_df) - 1

            # Update cyclical time features
            if 'hour_sin' in sim_df.columns:
                sim_df.loc[last_idx, 'hour_sin'] = np.sin(2 * np.pi * next_time.hour / 24)
            if 'hour_cos' in sim_df.columns:
                sim_df.loc[last_idx, 'hour_cos'] = np.cos(2 * np.pi * next_time.hour / 24)

            if 'day_of_week_sin' in sim_df.columns:
                sim_df.loc[last_idx, 'day_of_week_sin'] = np.sin(2 * np.pi * next_time.dayofweek / 7)
            if 'day_of_week_cos' in sim_df.columns:
                sim_df.loc[last_idx, 'day_of_week_cos'] = np.cos(2 * np.pi * next_time.dayofweek / 7)

            if 'month_sin' in sim_df.columns:
                sim_df.loc[last_idx, 'month_sin'] = np.sin(2 * np.pi * next_time.month / 12)
            if 'month_cos' in sim_df.columns:
                sim_df.loc[last_idx, 'month_cos'] = np.cos(2 * np.pi * next_time.month / 12)

            if 'is_rush_hour' in sim_df.columns:
                sim_df.loc[last_idx, 'is_rush_hour'] = 1 if next_time.hour in [7, 8, 9, 16, 17, 18, 19] else 0
            if 'is_workday' in sim_df.columns:
                sim_df.loc[last_idx, 'is_workday'] = 1 if next_time.dayofweek < 5 else 0

            # Update rolling features
            var_list = ['pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'pm10']
            for win in rolling_mean_windows:
                for vr in var_list:
                    col_name = f'{vr}_rolling_mean_{win}h'
                    if vr in sim_df.columns and col_name in sim_df.columns:
                        sim_df[col_name] = sim_df[vr].rolling(window=win).mean().shift(1)

            for win in rolling_std_windows:
                for vr in var_list:
                    col_name = f'{vr}_rolling_std_{win}h'
                    if vr in sim_df.columns and col_name in sim_df.columns:
                        sim_df[col_name] = sim_df[vr].rolling(window=win).std().shift(1)

            # Update derived features
            if 'pm_ratio' in sim_df.columns and 'pm2_5' in sim_df.columns and 'pm10' in sim_df.columns:
                sim_df['pm_ratio'] = np.where(sim_df['pm10'] > 0, sim_df['pm2_5'] / sim_df['pm10'], 0)

            if 'aqi_change_rate' in sim_df.columns:
                sim_df['aqi_change_rate'] = (sim_df[target_var].shift(1) - sim_df[target_var].shift(2)) / (sim_df[target_var].shift(2) + 1e-5)

            # Prepare feature vector - get last row and fill NaN values from previous rows
            try:
                last_row_features = sim_df[features].iloc[-1].copy()

                # Fill any missing values in features with forward fill then backward fill
                for feat in features:
                    if pd.isna(last_row_features[feat]):
                        # Try to get value from previous rows
                        valid_vals = sim_df[feat].dropna()
                        if len(valid_vals) > 0:
                            last_row_features[feat] = valid_vals.iloc[-1]

                # If still has NaN, skip this prediction
                if last_row_features.isna().any():
                    continue

                X_pred = pd.DataFrame([last_row_features])
                X_pred_scaled = scaler.transform(X_pred)

                # Predict
                pred_val = float(model.predict(X_pred_scaled)[0])
                sim_df.at[last_idx, target_var] = pred_val

                predictions.append({'datetime': next_time, 'predicted_us_aqi': pred_val})
            except (KeyError, IndexError) as e:
                print(f"Error preparing features in step {step}: {str(e)}")
                continue

        except Exception as e:
            print(f"Error in forecast step {step}: {str(e)}")
            continue

    return pd.DataFrame(predictions), sim_df


@app.get("/api/predict")
async def predict():
    """Get AQI prediction for next 96 hours or until end of day 3."""
    try:
        # Load model and metadata
        payload, metadata = load_latest_model(bucket_name="models")
        if payload is None:
            raise HTTPException(status_code=404, detail="No trained model found")

        model = payload['model']
        scaler = payload['scaler']
        features = payload['features']

        # Fetch latest features
        df = fetch_features(days=30, collection_name='features', timestamp_field='timestamp')
        if df.empty:
            raise HTTPException(status_code=404, detail="No features found")

        # Rename and prepare data
        if 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('Asia/Karachi').dt.tz_localize(None)
        df = df.sort_values('datetime', ascending=True).reset_index(drop=True)

        # Verify data exists
        if len(df) == 0:
            raise HTTPException(status_code=404, detail="No valid data after sorting")

        # Get current AQI (latest value)
        if 'us_aqi' not in df.columns:
            raise HTTPException(status_code=400, detail="'us_aqi' column not found in data")

        current_aqi = df['us_aqi'].iloc[-1]
        current_time = df['datetime'].iloc[-1]

        # Run forecast
        predictions_df, _ = forecast_96_hours(df, model, features, scaler)

        if predictions_df.empty:
            raise HTTPException(status_code=400, detail="Forecast produced no predictions")

        # Calculate end time: 3 days from today or after 96 hours
        today = pd.Timestamp.now().normalize()
        end_of_day_3 = today + pd.Timedelta(days=3, hours=23, minutes=59, seconds=59)

        # Filter predictions
        predictions_df = predictions_df[predictions_df['datetime'] <= end_of_day_3].reset_index(drop=True)

        # Group by day and get first hour of each day for daily predictions
        predictions_df['date'] = predictions_df['datetime'].dt.date
        daily_predictions = []

        for i, (date, group) in enumerate(predictions_df.groupby('date', sort=False)):
            if len(group) > 0:
                aqi_val = group['predicted_us_aqi'].iloc[0]
                daily_pred = {
                    'day': i,
                    'date': str(date),
                    'aqi': float(aqi_val),
                    'category': get_aqi_category(aqi_val)[0],
                    'color': get_aqi_category(aqi_val)[1]
                }
                daily_predictions.append(daily_pred)

        # Get last 24 hours of data
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=24)
        last_24h = df[df['datetime'] >= cutoff_time].sort_values('datetime')
        hourly_today = []

        if not last_24h.empty:
            for _, row in last_24h.iterrows():
                row_dict = {'datetime': row['datetime'].isoformat()}
                # Add all columns except datetime and derived features
                for col in row.index:
                    if col != 'datetime' and not col.startswith('_'):
                        val = row[col]
                        if pd.notna(val):
                            try:
                                row_dict[col] = float(val)
                            except (ValueError, TypeError):
                                pass
                hourly_today.append(row_dict)

        response = {
            'status': 'success',
            'current': {
                'aqi': float(current_aqi) if pd.notna(current_aqi) else None,
                'category': get_aqi_category(current_aqi)[0] if pd.notna(current_aqi) else None,
                'color': get_aqi_category(current_aqi)[1] if pd.notna(current_aqi) else None,
                'timestamp': current_time.isoformat()
            },
            'predictions': daily_predictions,
            'model_info': {
                'name': metadata.get('selected_model', 'Unknown'),
                'r2': metadata.get('selected_metrics', {}).get('r2'),
                'rmse': metadata.get('selected_metrics', {}).get('rmse'),
                'mae': metadata.get('selected_metrics', {}).get('mae')
            },
            'all_models': metadata.get('all_metrics', []),
            'hourly_today': hourly_today
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/api/shap-values")
async def shap_values():
    """Get pre-computed SHAP values for model explainability."""
    try:
        # Load model
        payload, metadata = load_latest_model(bucket_name="models")
        if payload is None:
            raise HTTPException(status_code=404, detail="No trained model found")

        shap_data = payload.get('shap_data')
        if shap_data is None:
            raise HTTPException(status_code=400, detail="SHAP data not available for current model")

        return shap_data

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"SHAP values error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve SHAP values: {str(e)}")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/debug")
async def debug():
    """Debug endpoint to check data and model availability."""
    try:
        # Check model
        payload, metadata = load_latest_model(bucket_name="models")
        model_status = "Model loaded successfully" if payload else "No model found"

        # Check features
        df = fetch_features(days=30, collection_name='features', timestamp_field='timestamp')
        features_status = f"Found {len(df)} feature rows" if not df.empty else "No features found"

        if not df.empty:
            df_info = {
                'rows': len(df),
                'columns': df.columns.tolist(),
                'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else "N/A"
            }
        else:
            df_info = {}

        return {
            "model": model_status,
            "features": features_status,
            "data_info": df_info,
            "metadata": metadata if payload is None else "✓ Valid"
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
