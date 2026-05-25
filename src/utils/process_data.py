import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process air quality data by handling missing values.

    Args:
        df: Input DataFrame with air quality features

    Returns:
        Processed DataFrame with missing values handled
    """

    df = df.ffill(limit=2)
    df = df.interpolate(method='linear', limit_direction='both')
    df = df.bfill().ffill()

    return df


def iqr_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers using IQR method.

    Args:
        df: Input DataFrame with air quality features

    Returns:
        DataFrame with outliers capped at IQR bounds
    """
    feature_cols = df.columns.to_list()

    df_capped = df.copy()

    for col in feature_cols:
        if col in ['wind_direction_10m', 'cloud_cover', 'relative_humidity_2m']:
            continue
        Q1 = df_capped[col].quantile(0.25)
        Q3 = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_capped[col] = np.clip(df_capped[col], lower_bound, upper_bound)

    return df_capped


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to dataframe

    - Time-based features (hour, day_of_week, month) with cyclical encodings
    - Wind components (`wind_u`, `wind_v`) from speed & direction
    - Rolling means and standard deviations for select variables
    - Derived features: `aqi_change_rate`, `pm_ratio`, `is_rush_hour`,
      `is_workday`, and `precip_likelihood`
    """
    df = df.copy()

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])

    # Time-based features
    if 'datetime' in df.columns:
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)

        df = df.drop(columns=[c for c in ['hour', 'day_of_week', 'month'] if c in df.columns])


    if 'wind_speed_10m' in df.columns and 'wind_direction_10m' in df.columns:
        rad = np.deg2rad(df['wind_direction_10m'])
        df['wind_u'] = -df['wind_speed_10m'] * np.sin(rad)
        df['wind_v'] = -df['wind_speed_10m'] * np.cos(rad)

        df = df.drop(columns=['wind_direction_10m'])

    # Rolling means and standard deviations
    rolling_mean_windows = [36]
    rolling_std_windows = [72]
    rolling_variables = ['pm2_5', 'carbon_monoxide', 'pm10']

    for window in rolling_mean_windows:
        for var in rolling_variables:
            if var in df.columns:
                df[f'{var}_rolling_mean_{window}h'] = (
                    df[var].shift(1).rolling(window=window, min_periods=1).mean()
                )

    for window in rolling_std_windows:
        for var in rolling_variables:
            if var in df.columns and var != 'carbon_monoxide':
                df[f'{var}_rolling_std_{window}h'] = (
                    df[var].shift(1).rolling(window=window, min_periods=1).std()
                )

    df = df.ffill()
    df = df.dropna()

    if 'us_aqi' in df.columns:
        df['aqi_change_rate'] = df['us_aqi'].diff()
    else:
        df['aqi_change_rate'] = np.nan

    if 'pm2_5' in df.columns and 'pm10' in df.columns:
        # avoid division by zero
        df['pm_ratio'] = np.where(df['pm10'] > 0, df['pm2_5'] / df['pm10'], 0)
    else:
        df['pm_ratio'] = 0

    if 'datetime' in df.columns:
        hour = df['datetime'].dt.hour
        df['is_rush_hour'] = (((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))).astype(int)
    else:
        df['is_rush_hour'] = 0

    if 'relative_humidity_2m' in df.columns and 'cloud_cover' in df.columns:
        df['precip_likelihood'] = df['relative_humidity_2m'] * df['cloud_cover']
    else:
        df['precip_likelihood'] = 0

    # Final fill for any remaining small gaps
    df = df.bfill().ffill()

    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete data preparation pipeline.

    Steps:
    1. Clean data (handle missing values)
    2. Winsorize outliers
    3. Engineer features

    Args:
        df: Raw input DataFrame

    Returns:
        Fully processed and engineered DataFrame ready for modeling
    """
    df = clean_data(df)
    df = iqr_data(df)
    df = engineer_features(df)

    return df
