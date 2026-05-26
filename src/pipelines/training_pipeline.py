import pickle
import pandas as pd
import numpy as np
from typing import Optional
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

    # remove future rows and sort ascending
    now = pd.Timestamp.now()
    df = df[df['datetime'] < now].sort_values('datetime', ascending=True).reset_index(drop=True)

    if df.empty:
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

    # Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_metrics = []
    models = {}

    # 1. Random Forest
    try:
        rf = RandomForestRegressor(n_estimators=200, max_depth=15, max_features=0.7,
                                   min_samples_split=5, min_samples_leaf=4, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        rf_pred = rf.predict(X_test_scaled)
        model_metrics.append(calculate_metrics(rf_pred, y_test, 'RandomForest'))
        models['RandomForest'] = rf
    except Exception:
        pass

    # 2. Gradient Boosting
    try:
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.5, max_depth=3,
                                       min_samples_split=15, min_samples_leaf=5, subsample=0.7,
                                       random_state=42)
        gb.fit(X_train_scaled, y_train)
        gb_pred = gb.predict(X_test_scaled)
        model_metrics.append(calculate_metrics(gb_pred, y_test, 'GradientBoosting'))
        models['GradientBoosting'] = gb
    except Exception:
        pass

    # 3. XGBoost
    if XGBRegressor is not None:
        try:
            xgb = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                               gamma=0.5, random_state=42, n_jobs=-1)
            xgb.fit(X_train_scaled, y_train)
            xgb_pred = xgb.predict(X_test_scaled)
            model_metrics.append(calculate_metrics(xgb_pred, y_test, 'XGBoost'))
            models['XGBoost'] = xgb
        except Exception:
            pass

    # 4. SVR
    try:
        svr = SVR(kernel='rbf', C=10, gamma=0.001, epsilon=0.1)
        svr.fit(X_train_scaled, y_train)
        svr_pred = svr.predict(X_test_scaled)
        model_metrics.append(calculate_metrics(svr_pred, y_test, 'SVR'))
        models['SVR'] = svr
    except Exception:
        pass

    # 5. Ridge
    try:
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        ridge_pred = ridge.predict(X_test_scaled)
        model_metrics.append(calculate_metrics(ridge_pred, y_test, 'Ridge'))
        models['Ridge'] = ridge
    except Exception:
        pass

    if not model_metrics:
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

    # Serialize model + scaler together so we can reproduce preprocessing at inference
    payload = {'model': best_model, 'scaler': scaler, 'features': X.columns.tolist()}
    try:
        model_bytes = pickle.dumps(payload)
    except Exception as e:
        raise RuntimeError("Error while trying to pickle best model") from e

    try:
        file_id = store_model_pickle(model_bytes=model_bytes, model_name=best_name, metadata=metadata)
    except Exception as e:
        raise RuntimeError("Error while trying to store best model in DB") from e

    return file_id


if __name__ == '__main__':
    fid = training_pipeline()
    print('Stored model file id:', fid)
