import os

import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables from .env (if present)
load_dotenv()


def store_engineered_features(
	df: pd.DataFrame,
	collection_name: str = "features",
	timestamp_col: str = "datetime",
) -> int:
	"""Ingest engineered features from a DataFrame into MongoDB.

	`df` should already be a pandas DataFrame. Function reads `MONGO_URI`
	and `DB_NAME` from the environment and inserts simplified documents.
	Returns the number of documents inserted.
	"""

	if not isinstance(df, pd.DataFrame):
		raise TypeError("df must be a pandas DataFrame")

	# Read env and connect
	mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
	db_name = os.getenv("DB_NAME", "ml_platform")
	client = MongoClient(mongo_uri)
	db = client[db_name]
	feature_collection = db[collection_name]

	# Ensure index on timestamp (descending)
	feature_collection.create_index([("timestamp", -1)], unique=True)

	if timestamp_col not in df.columns:
		raise ValueError(f"DataFrame is missing required column: {timestamp_col}")

	# Convert timestamps and sanitize NaNs
	df = df.copy()
	df[timestamp_col] = pd.to_datetime(df[timestamp_col])
	df = df.where(pd.notnull(df), None)

	# Build documents
	mongo_documents = []
	for _, row in df.iterrows():
		ts = row[timestamp_col]
		if hasattr(ts, "to_pydatetime"):
			ts = ts.to_pydatetime()

		features = row.drop(labels=[timestamp_col]).to_dict()
		doc = {"timestamp": ts, "features": features}
		mongo_documents.append(doc)

	# Insert
	if not mongo_documents:
		return 0

	result = feature_collection.insert_many(mongo_documents)
	return len(result.inserted_ids)


def fetch_features(days=None, collection_name: str = "features", timestamp_field: str = "datetime") -> pd.DataFrame:
	"""Fetch features from the feature store.

	If `days` is provided (int), fetch documents with `timestamp >= now - days`.
	If `days` is None, fetch all documents. Returns a pandas DataFrame with a
	`timestamp` column and the flattened feature columns.
	"""

	# Read env and connect
	mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
	db_name = os.getenv("DB_NAME", "ml_platform")
	client = MongoClient(mongo_uri)
	db = client[db_name]
	feature_collection = db[collection_name]

	# Build query
	query = {}
	if days is not None:
		if not isinstance(days, int) or days < 0:
			raise ValueError("days must be a non-negative integer or None")
		cutoff = datetime.now() - timedelta(days=days)
		query = {timestamp_field: {"$gte": cutoff}}

	# Fetch documents sorted by timestamp descending
	cursor = feature_collection.find(query).sort(timestamp_field, -1)

	docs = list(cursor)
	if not docs:
		return pd.DataFrame()

	# Flatten documents: each doc -> {timestamp: ..., **features}
	records = []
	for d in docs:
		rec = {timestamp_field: d.get(timestamp_field)}
		features = d.get("features") or {}
		if isinstance(features, dict):
			rec.update(features)
		else:
			rec["features"] = features
		records.append(rec)

	df = pd.DataFrame.from_records(records)
	return df

