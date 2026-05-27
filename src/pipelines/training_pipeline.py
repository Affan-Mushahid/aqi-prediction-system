import pickle
import pandas as pd
import numpy as np
from typing import Optional
from datetime import timezone
from src.utils.db_utils import fetch_features, store_model_pickle
from src.utils.process_data import prepare_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


def calculate_metrics(y_pred, y_true, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'name': model_name, 'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}


def initialize_random_forest():
    try:
        return RandomForestRegressor(n_estimators=200, max_depth=15, max_features=0.7,
                                     min_samples_split=5, min_samples_leaf=4, random_state=42, n_jobs=-1)
    except Exception:
        return None


def initialize_gradient_boosting():
    try:
        return GradientBoostingRegressor(n_estimators=100, learning_rate=0.5, max_depth=3,
                                         min_samples_split=15, min_samples_leaf=5, subsample=0.7,
                                         random_state=42)
    except Exception:
        return None


def initialize_xgboost():
    if XGBRegressor is None:
        return None
    try:
        return XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                            gamma=0.5, random_state=42, n_jobs=-1)
    except Exception:
        return None


def initialize_svr():
    try:
        return SVR(kernel='rbf', C=10, gamma=0.001, epsilon=0.1)
    except Exception:
        return None


def initialize_ridge():
    try:
        return Ridge(alpha=1.0)
    except Exception:
        return None


def prepare_and_scale(X_train, X_test, scaler: Optional[MinMaxScaler] = None):
    """Fit (or reuse) a scaler on X_train and transform X_train and X_test.

    Returns: scaler, X_train_scaled, X_test_scaled
    """
    if scaler is None:
        scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return scaler, X_train_scaled, X_test_scaled


def training_pipeline() -> Optional[object]:
    """Fetch features from DB, train multiple regressors, pick best model and store it.

    Returns the GridFS file id of the stored model on success, or None on failure/no-op.
    """

    df = fetch_features(days=None, collection_name='features', timestamp_field='timestamp')


    if df.empty:
        raise ValueError("Database has no features to train model on")


    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Normalize dataframe datetimes to timezone-aware UTC
    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    else:
        df['datetime'] = df['datetime'].dt.tz_convert('UTC')

    # remove future rows and sort ascending using UTC now
    now = pd.Timestamp.now(tz=timezone.utc)
    df = df[df['datetime'] < now].sort_values('datetime', ascending=True).reset_index(drop=True)

    if df.empty:
        print("No historical rows after filtering out future timestamps; nothing to train.")
        return None

    # Create target
    if 'us_aqi' not in df.columns:
        raise ValueError("Critical features are missing from feature store. Check for data corruption")
    df['target'] = df['us_aqi'].shift(-1)
    df = df.ffill()
    df = df.dropna()
    df = df.bfill()

    # Train-test split
    n = len(df)
    split_idx = int(n * 0.8)

    drop_cols = ['datetime', 'target', 'us_aqi']
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(cols_to_drop, axis=1)
    y = df['target']

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    # Scale features using train/test splits (splitting preserved above)
    scaler, X_train_scaled, X_test_scaled = prepare_and_scale(X_train, X_test)

    model_metrics = []
    models = {}

    # Initialize models via per-model functions
    for name, fn in [
        ('RandomForest', initialize_random_forest),
        ('GradientBoosting', initialize_gradient_boosting),
        ('XGBoost', initialize_xgboost),
        ('SVR', initialize_svr),
        ('Ridge', initialize_ridge),
    ]:
        try:
            mdl = fn()
            if mdl is not None:
                models[name] = mdl
        except Exception:
            continue

    # Train each initialized model and collect metrics; remove models that fail to fit
    for name, mdl in list(models.items()):
        try:
            mdl.fit(X_train_scaled, y_train)
            preds = mdl.predict(X_test_scaled)
            model_metrics.append(calculate_metrics(preds, y_test, name))
            models[name] = mdl
        except Exception:
            models.pop(name, None)

    if not model_metrics:
        print("No models were successfully trained; aborting and returning None.")
        return None

    metrics_df = pd.DataFrame(model_metrics)

    # Select best model: highest r2, then lowest rmse, then lowest mae
    metrics_df_sorted = metrics_df.sort_values(by=['r2', 'rmse', 'mae'], ascending=[False, True, True]).reset_index(drop=True)
    best_row = metrics_df_sorted.iloc[0]
    best_name = best_row['name']
    best_model = models.get(best_name)
    if best_model is None:
        raise ValueError("Can't select best model. No model to select")

    
    # Prepare metadata: include chosen metrics and all models' metrics
    all_metrics = metrics_df.to_dict(orient='records')
    metadata = {
        'selected_model': best_name,
        'selected_metrics': {'r2': float(best_row['r2']), 'rmse': float(best_row['rmse']), 'mae': float(best_row['mae'])},
        'all_metrics': all_metrics,
        'features': X.columns.tolist()
    }

    # Retrain selected model on the entire dataset with a fresh scaler
    scaler = MinMaxScaler()
    X_full_scaled = scaler.fit_transform(X)

    # Re-initialize the best model (fresh instance) before fitting on full data
    initializer_map = {
        'RandomForest': initialize_random_forest,
        'GradientBoosting': initialize_gradient_boosting,
        'XGBoost': initialize_xgboost,
        'SVR': initialize_svr,
        'Ridge': initialize_ridge,
    }
    init_fn = initializer_map.get(best_name)
    if init_fn is None:
        raise ValueError(f"No initializer found for selected model '{best_name}'")
    fresh_model = init_fn()
    if fresh_model is None:
        raise RuntimeError(f"Failed to initialize a fresh instance of selected model '{best_name}'")

    try:
        fresh_model.fit(X_full_scaled, y)
        best_model = fresh_model
    except Exception as e:
        raise RuntimeError("Error retraining the selected model on full dataset") from e

    # Update metadata to indicate model was retrained on full dataset
    metadata.update({'retrained_on_full': True, 'trained_samples': int(len(X))})

    # Serialize retrained model + scaler together so we can reproduce preprocessing at inference
    payload = {'model': best_model, 'scaler': scaler, 'features': X.columns.tolist()}
    try:
        model_bytes = pickle.dumps(payload)
    except Exception as e:
        raise RuntimeError("Error while trying to pickle retrained best model") from e

    try:
        file_id = store_model_pickle(model_bytes=model_bytes, model_name=best_name, metadata=metadata)
    except Exception as e:
        raise RuntimeError("Error while trying to store retrained model in DB") from e

    return file_id


if __name__ == '__main__':
    fid = training_pipeline()
    print('Stored model file id:', fid)
