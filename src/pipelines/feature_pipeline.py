import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from src.utils.process_data import prepare_data
from src.utils.db_utils import fetch_features, store_engineered_features


def fetch_hourly_features() -> Optional[int]:
    latest_df = fetch_features(days=14, collection_name='features', timestamp_field='timestamp')

    if latest_df.empty:
        # nothing to compare against — exit quietly
        raise ValueError("Run Backfill features first")

    latest_row = latest_df.iloc[0]
    latest_ts = pd.to_datetime(latest_row.get('timestamp'))
    if pd.isna(latest_ts):
        raise ValueError("Could not fetch latest record's date for comparing. Check for corruption")

    # Fetch API: last 7 days up to current hour - 1
    now_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
    end_dt = now_hour - timedelta(hours=1)
    start_dt = (end_dt - timedelta(days=7)).date()

    if start_dt is None or end_dt is None or start_dt > end_dt.date():
        return 0

    features = [
        'pm2_5', 'pm10', 'carbon_monoxide', 'carbon_dioxide',
        'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'dust', 'uv_index'
    ]
    weather_features = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'wind_direction_10m', 'cloud_cover']

    latitude, longitude = 24.8607, 67.0011

    aq_params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_dt.strftime('%Y-%m-%d'),
        'end_date': end_dt.strftime('%Y-%m-%d'),
        'hourly': ','.join(features + ['us_aqi']),
        'timezone': 'Asia/Karachi'
    }
    aq_url = 'https://air-quality-api.open-meteo.com/v1/air-quality'
    try:
        resp = requests.get(aq_url, params=aq_params)
        resp.raise_for_status()
        aq_data = resp.json()
    except Exception as e:
        print("Open-Meteo API: AQI API is down.", e)
        return 0

    times = aq_data.get('hourly', {}).get('time')
    if not times:
        return 0

    aq_df = pd.DataFrame({'datetime': pd.date_range(start=times[0], periods=len(times), freq='h')})
    for f in features:
        aq_df[f] = aq_data['hourly'].get(f)
    aq_df['us_aqi'] = aq_data['hourly'].get('us_aqi')

    aq_df['datetime'] = pd.to_datetime(aq_df['datetime'])
    # ensure we don't include future hours
    aq_df = aq_df[aq_df['datetime'] <= end_dt].reset_index(drop=True)
    if aq_df.empty:
        return 0


    weather_params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_dt.strftime('%Y-%m-%d'),
        'end_date': end_dt.strftime('%Y-%m-%d'),
        'hourly': ','.join(weather_features),
        'timezone': 'Asia/Karachi'
    }
    weather_url = 'https://api.open-meteo.com/v1/forecast'
    try:
        resp = requests.get(weather_url, params=weather_params)
        resp.raise_for_status()
        weather = resp.json()
    except Exception as e:
        print("Open-Meteo API: Weather API is down.", e)
        return 0

    wf_times = weather.get('hourly', {}).get('time')
    if not wf_times:
        return 0

    weather_df = pd.DataFrame({'datetime': pd.date_range(start=wf_times[0], periods=len(wf_times), freq='h')})
    for wf in weather_features:
        weather_df[wf] = weather['hourly'].get(wf)

    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
    weather_df = weather_df[weather_df['datetime'] <= end_dt].reset_index(drop=True)

    # Combine: similar to backfill_aqi_features
    merged = pd.merge(aq_df, weather_df, on='datetime', how='left')
    w_cols = [c for c in weather_features if c in merged.columns]
    if w_cols:
        merged = merged.set_index('datetime')
        merged[w_cols] = merged[w_cols].interpolate(method='time', limit_direction='both')
        merged = merged.reset_index()
        merged[w_cols] = merged[w_cols].ffill().bfill()

    # Cut off rows beyond latest AQ time and beyond end_dt
    merged = merged[merged['datetime'] <= end_dt].reset_index(drop=True)
    if merged.empty:
        return 0


    prepared = prepare_data(merged)


    latest_df_norm = latest_df.copy()
    if 'timestamp' in latest_df_norm.columns:
        latest_df_norm = latest_df_norm.rename(columns={'timestamp': 'datetime'})
    if 'datetime' in latest_df_norm.columns:
        latest_df_norm['datetime'] = pd.to_datetime(latest_df_norm['datetime'])

    # Remove rows that are <= latest DB datetime
    new_rows = prepared[prepared['datetime'] > latest_ts].reset_index(drop=True)
    if new_rows.empty:
        return 0

    # Compare each new row to the latest DB row: if any row has ALL features
    # (excluding datetime) exactly equal to latest DB row, exit without storing.
    # Determine common columns to compare (exclude datetime)
    latest_features = latest_df_norm.iloc[0].drop(labels=['datetime'], errors='ignore')
    compare_cols = [c for c in new_rows.columns if c != 'datetime' and c in latest_features.index]

    for _, r in new_rows.iterrows():
        equal = True
        for col in compare_cols:
            a = latest_features.get(col)
            b = r.get(col)
            if pd.isna(a) and pd.isna(b):
                continue
            if a != b:
                equal = False
                break
        if equal:
            return 0

    inserted = store_engineered_features(new_rows)

    return inserted


if __name__ == '__main__':
    inserted = fetch_hourly_features()
    print('Inserted:', inserted)
