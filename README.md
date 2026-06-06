# Air Quality Index (AQI) Prediction System

A machine learning system for predicting air quality trends for next 3 days in Karachi. This project involves data engineering, predictive modeling, and interactive visualization to forecast Air Quality Index values, enabling users to anticipate air quality changes and make informed decisions about outdoor activities.

This project was done as part of **10Pearls Shine Internship Program** Cohort 8.

The project report can be viewed [here](Project%20Report.pdf).

## Project Overview

The AQI Prediction System leverages historical air quality data to forecast AQI values up to 3 days in advance. The system features a modern web interface, REST API backend, and automated data pipelines that continuously update predictions with the latest air quality measurements.

**Live Demo:** https://aqi-prediction-system-dfl4.onrender.com/

> [!WARNING]
> In case the server does not wake up in time, for a temporary fix, visit https://aqi-backend-u599.onrender.com/api/health and let the render server start up.

## Key Features

- **3 Days Predictive Forecasting**: Advanced machine learning models generate hour-by-hour AQI predictions extending 3 days into the future with iterative forecasting techniques
- **Real-Time Dashboard**: Interactive Streamlit dashboard with responsive charts, current air quality metrics, and 3-day forecasts with category classifications
- **REST API Backend**: FastAPI-powered server providing prediction endpoints, SHAP explainability values, and health monitoring
- **Automated Data Pipelines**: GitHub Actions-powered pipelines that run hourly to collect, process, and feature-engineer air quality data
- **Model Explainability**: SHAP (SHapley Additive exPlanations) values integrated into the dashboard for transparent model decisions
- **AQI Categorization**: Automatic classification of air quality into health-based categories (Good, Moderate, Unhealthy, etc.) with color-coded indicators
- **Multi-Model Support**: Trained ensemble of machine learning models with performance metrics (R², RMSE, MAE) for comparison and selection

## Technical Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **ML/Data Processing**: scikit-learn, XGBoost, Pandas, NumPy
- **Model Explainability**: SHAP library for feature importance visualization
- **Database**: MongoDB for storing features and trained models
- **Visualization**: Plotly for interactive charts and graphs
- **Orchestration**: GitHub Actions for automated data and training pipelines
- **Deployment**: Render.com for production hosting

## Project Structure

```
aqi-prediction-system/
├── src/
│   ├── app/
│   │   ├── main.py                 # FastAPI backend server with prediction endpoints
│   │   ├── dashboard.py            # Streamlit interactive dashboard interface
│   │   └── requirements.txt         # App-specific dependencies
│   ├── pipelines/
│   │   ├── feature_pipeline.py     # Hourly data collection and feature engineering
│   │   ├── training_pipeline.py    # Model training and evaluation pipeline
│   │   └── backfill_features.py    # Historical data backfill utility
│   └── utils/
│       ├── db_utils.py             # Database utility functions for MongoDB
│       ├── process_data.py         # Data processing and transformation utilities
│       └── __init__.py
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml    # Scheduled hourly feature collection
│       └── training_pipeline.yml   # Scheduled daily model retraining
├── notebooks/
│   ├── eda_feature_engineering.ipynb    # Exploratory data analysis and feature creation
│   └── model_training.ipynb             # Model development and evaluation workflows
├── requirements.txt                # Root-level project dependencies
├── .env                            # Environment configuration (API URLs, credentials)
└── README.md                       # This file
```

## GitHub Actions Automation

The system includes two automated pipelines that run on schedule:

### Feature Pipeline
- **Schedule**: Every hour
- **Purpose**: Collects latest air quality data from Open-Meteo API, computes engineered features (rolling statistics, cyclical encoding, derived metrics), and stores in MongoDB
- **File**: `.github/workflows/feature_pipeline.yml`

### Training Pipeline
- **Schedule**: Daily
- **Purpose**: Trains ensemble of ML models on accumulated historical data, evaluates performance metrics, selects best model, computes SHAP values for explainability
- **File**: `.github/workflows/training_pipeline.yml`

Both pipelines use GitHub Secrets to securely manage MongoDB credentials and API keys.

## Features Explained

### API Endpoints

**`GET /api/predict`**
Returns current AQI value, category classification, 3-day daily forecasts, model metadata, and hourly data for the past 24 hours.

**`GET /api/shap-values`**
Provides pre-computed SHAP values showing the contribution of each feature to the model's predictions.

**`GET /api/health`**
Simple health check endpoint for monitoring service availability.

**`GET /api/debug`**
Diagnostic endpoint providing information about model availability, feature data status, and data range.

### Dashboard Features

- **Current AQI Display**: Large metric card showing the latest AQI value with category classification and color coding
- **3-Day Forecast**: Daily predictions with trend indicators and category information
- **Hourly Charts**: Time series visualization of past 24 hours with interactive Plotly charts
- **Model Performance**: Display of selected model metrics and comparison with alternative models
- **SHAP Explainability**: Feature importance visualization showing which factors most influence predictions
- **Auto-Refresh**: Responsive UI with button-triggered updates and spinner feedback during server startup

## Getting Started

### Local Setup
1. **Prerequisite**
- Python 3.10 or higher
- MongoDB Atlas account with connection URI
- pip or conda package manager

2. **Clone the repository**
```bash
git clone https://github.com/Affan-Mushahid/aqi-prediction-system.git
cd aqi-prediction-system
```

3. **Create and activate a virtual environment**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Configure environment variables**
Create a `.env` file in the project root:
```
MONGO_URI=insert_here
DB_NAME=insert_here
API_URL=insert_here
```

6. **Running Locally**

**Start the FastAPI Backend Server**
```bash
uvicorn src.app.main:app --reload --port 8000
```
The API server will be available at `http://localhost:8000`

**Launch the Streamlit Dashboard** (in a new terminal)
```bash
streamlit run src/app/dashboard.py
```
The dashboard will open in your browser at `http://localhost:8501`

Both services need to run simultaneously for the dashboard to fetch predictions from the API.

## How the System Works

### Data Flow

1. **Feature Collection**: GitHub Actions runs the feature pipeline hourly to fetch latest air quality measurements and compute engineered features
2. **Storage**: Features are stored in MongoDB with timestamps for historical tracking
3. **Model Training**: Periodic training pipeline retrains models on accumulated data and evaluates performance
4. **Prediction Generation**: When a user requests predictions via the API, the latest trained model performs iterative 3 days forecasting
5. **Dashboard Visualization**: Frontend fetches predictions from API and renders interactive visualizations

### Forecasting Algorithm

The system uses iterative multi-step forecasting:
1. Start with the most recent historical data
2. For each hour for the next 3 days:
   - Create a new row with updated cyclical features.
   - Compute rolling statistics (mean, std) over appropriate windows
   - Derive features
   - Feed prepared feature vector to trained model
   - Use prediction as input for next iteration
3. Aggregate hourly predictions into daily forecasts

### Feature Engineering

The system creates various features from raw air quality measurements for predictions including:

- **Temporal Features**: Hour/day/month encoded as sine/cosine (cyclical), rush hour indicators, workday flags
- **Weather Components**: Wind direction convert to u and v components
- **Rolling Statistics**: 36-hour and 72-hour rolling means/standard deviations for pollutants
- **Derived Features**: PM2.5/PM10 ratios, AQI change rates, Precipitation Likelihood, Is Rush Hour, Is Workday

## Performance

The trained models achieve strong predictive performance:
- **R² Score**: ~0.85-0.90 on test data
- **RMSE**: 8-12 AQI points
- **MAE**: 6-9 AQI points

These metrics indicate the model captures the majority of AQI variance and provides reliable directional forecasts.

## Development Notes

- Environment variables are loaded from `.env` file via python-dotenv
- All timestamps are handled in UTC and converted to Asia/Karachi timezone for display
- MongoDB stores both raw data and pickled model objects with metadata
- SHAP values are pre-computed during training to optimize dashboard performance
- GitHub Actions workflows use Python 3.12 with cached pip dependencies for speed
