import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from src.utils.process_data import prepare_data
from src.utils.db_utils import store_engineered_features, clear_database


def backfill_aqi_features(days=365):
    features = ['pm2_5', 'pm10', 'carbon_monoxide', 'carbon_dioxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'dust', 'uv_index']
    weather_features = ['temperature_2m','relative_humidity_2m','wind_speed_10m','wind_direction_10m', 'cloud_cover']

    # Location (Karachi)
    latitude, longitude = 24.8607, 67.0011

    end_date_aq = datetime.now(timezone.utc).date()
    start_date = end_date_aq - timedelta(days=days)

    aq_params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date_aq.strftime('%Y-%m-%d'),
        'hourly': ','.join(features + ['us_aqi']),
        'timezone': 'UTC'
    }

    aq_url = 'https://air-quality-api.open-meteo.com/v1/air-quality'
    resp = requests.get(aq_url, params=aq_params)
    resp.raise_for_status()
    aq_data = resp.json()

    try:
        times = aq_data['hourly']['time']
    except KeyError:
        raise RuntimeError('Unexpected air-quality response structure')

    aq_df = pd.DataFrame({'datetime': pd.date_range(start=times[0], periods=len(times), freq='h', tz='UTC')})
    for f in features:
        aq_df[f] = aq_data['hourly'].get(f)
    aq_df['us_aqi'] = aq_data['hourly'].get('us_aqi')

    now_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    aq_df['datetime'] = pd.to_datetime(aq_df['datetime'])
    if aq_df['datetime'].dt.tz is None:
        aq_df['datetime'] = aq_df['datetime'].dt.tz_localize('UTC')
    else:
        aq_df['datetime'] = aq_df['datetime'].dt.tz_convert('UTC')
    aq_df = aq_df[aq_df['datetime'] <= now_hour].reset_index(drop=True)
    if aq_df.empty:
        raise RuntimeError('No air-quality data found for the requested period up to current time')

    # WEATHER: 1) fetch historical archive up to yesterday (end_date - 1)
    # 2) fetch forecast for last day and append to ensure latest hour coverage
    end_date_hist = end_date_aq - timedelta(days=1)
    weather_params_hist = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date_hist.strftime('%Y-%m-%d'),
        'hourly': ','.join(weather_features),
        'timezone': 'UTC'
    }

    weather_url_hist = 'https://archive-api.open-meteo.com/v1/archive'
    resp = requests.get(weather_url_hist, params=weather_params_hist)
    resp.raise_for_status()
    weather_hist = resp.json()

    try:
        w_times = weather_hist['hourly']['time']
    except KeyError:
        raise RuntimeError('Unexpected weather-archive response structure')

    weather_df = pd.DataFrame({'datetime': pd.date_range(start=w_times[0], periods=len(w_times), freq='h', tz='UTC')})
    for wf in weather_features:
        weather_df[wf] = weather_hist['hourly'].get(wf)

    # Fetch forecast for the last day (to cover most recent hours)
    weather_params_fc = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': ','.join(weather_features),
        'start_date': end_date_aq.strftime('%Y-%m-%d'),
        'end_date': end_date_aq.strftime('%Y-%m-%d'),
        'timezone': 'UTC'
    }
    weather_url_fc = 'https://api.open-meteo.com/v1/forecast'
    resp = requests.get(weather_url_fc, params=weather_params_fc)
    resp.raise_for_status()
    weather_fc = resp.json()

    try:
        wf_times = weather_fc['hourly']['time']
    except KeyError:
        raise RuntimeError('Unexpected weather-forecast response structure')

    weather_fc_df = pd.DataFrame({'datetime': pd.date_range(start=wf_times[0], periods=len(wf_times), freq='h', tz='UTC')})
    for wf in weather_features:
        weather_fc_df[wf] = weather_fc['hourly'].get(wf)

    # Combine historical + forecast weather, drop duplicates, ensure continuous hourly coverage
    weather_all = pd.concat([weather_df, weather_fc_df], ignore_index=True)
    weather_all['datetime'] = pd.to_datetime(weather_all['datetime'])
    if weather_all['datetime'].dt.tz is None:
        weather_all['datetime'] = weather_all['datetime'].dt.tz_localize('UTC')
    else:
        weather_all['datetime'] = weather_all['datetime'].dt.tz_convert('UTC')
    weather_all = weather_all.drop_duplicates(subset=['datetime']).sort_values('datetime').reset_index(drop=True)

    latest_aq_time = aq_df['datetime'].max()
    weather_end = min(weather_all['datetime'].max(), latest_aq_time)
    full_idx = pd.date_range(start=weather_all['datetime'].min(), end=weather_end, freq='h', tz='UTC')
    weather_all = weather_all.set_index('datetime').reindex(full_idx).rename_axis('datetime').reset_index()

    num_cols = [c for c in weather_all.columns if c != 'datetime']
    weather_all = weather_all.set_index('datetime')
    weather_all[num_cols] = weather_all[num_cols].interpolate(method='time', limit_direction='both')
    weather_all = weather_all.reset_index()
    weather_all[num_cols] = weather_all[num_cols].ffill().bfill()

    merged = pd.merge(aq_df, weather_all, on='datetime', how='left')

    w_cols = [c for c in weather_features if c in merged.columns]
    if w_cols:
        merged = merged.set_index('datetime')
        merged[w_cols] = merged[w_cols].interpolate(method='time', limit_direction='both')
        merged = merged.reset_index()
        merged[w_cols] = merged[w_cols].ffill().bfill()

    # Cut off any rows beyond the latest air-quality datetime (to avoid empty rows)
    latest_aq_time = aq_df['datetime'].max()
    merged = merged[merged['datetime'] <= latest_aq_time].reset_index(drop=True)

    return merged


if __name__ == '__main__':
    cleared = clear_database()
    df = backfill_aqi_features()
    df = prepare_data(df)
    store_engineered_features(df)
    print(f"Cleared Database: ")
    print(f"Features deleted: {cleared['features_deleted']}, Models deleted: {'models_deleted'}")
    print(f"Processed {len(df)} records and stored to feature store")
    print(df.head())