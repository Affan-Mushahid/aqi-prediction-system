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
	mongo_uri = os.environ["MONGO_URI"]
	db_name = os.environ["DB_NAME"]
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

	# Read env and connect (use sensible defaults when env vars are missing)
	mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
	db_name = os.environ.get("DB_NAME", "ml_platform")
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


def store_model_pickle(
	model_bytes: bytes,
	model_name: str,
	metadata: dict = None,
	filename: str = None,
	bucket_name: str = "models",
) -> object:
	"""Store a pickle (as bytes) into GridFS and return the file id.

	`model_bytes` should be bytes produced by `pickle.dumps(model)`.
	`model_name` is a required identifier used to override previous models
	with the same name. `filename` is optional; if omitted a timestamped
	name will be used.
	"""

	if not isinstance(model_bytes, (bytes, bytearray)):
		raise TypeError("model_bytes must be bytes or bytearray")

	if not model_name or not isinstance(model_name, str):
		raise ValueError("model_name must be a non-empty string")

	if not filename:
		filename = f"{model_name}_{datetime.now().strftime('%Y%m%dT%H%M%SZ')}.pkl"

	# Try to infer model type/class from the pickle
	import pickle
	inferred_type = None
	try:
		obj = pickle.loads(model_bytes)
		inferred_type = obj.__class__.__name__
	except Exception:
		inferred_type = None

	meta = dict(metadata) if metadata else {}
	# store both logical name and inferred type
	meta["model_name"] = model_name
	meta.setdefault("model_type", inferred_type or "unknown")
	meta.setdefault("uploaded_at", datetime.now())

	# Read env and connect (use sensible defaults when env vars are missing)
	mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
	db_name = os.environ.get("DB_NAME", "ml_platform")
	client = MongoClient(mongo_uri)
	db = client[db_name]

	# Use GridFS with the specified bucket/collection prefix
	import gridfs
	fs = gridfs.GridFS(db, collection=bucket_name)

	# Remove all existing models in this bucket so the registry holds only one model.
	try:
		files_coll = db[f"{bucket_name}.files"]
		existing = list(files_coll.find({}))
		for ef in existing:
			try:
				fs.delete(ef["_id"])
			except Exception:
				pass
	except Exception:
		# If deletion fails, proceed to store the new model anyway.
		pass

	file_id = fs.put(model_bytes, filename=filename, metadata=meta)
	return file_id


def load_latest_model(bucket_name: str = "models"):
	"""Retrieve the latest model from the registry and return a tuple
	`(model_object, metadata)`.

	Ignores `model_name` and simply returns the most recently uploaded
	file from the specified GridFS bucket. Returns `(None, None)` if no
	model is found. If the file is present but unpickling fails, the
	metadata will still be returned alongside `None` for the model.
	"""
	# Read env and connect (use sensible defaults when env vars are missing)
	mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
	db_name = os.environ.get("DB_NAME", "ml_platform")
	client = MongoClient(mongo_uri)
	db = client[db_name]

	import gridfs
	fs = gridfs.GridFS(db, collection=bucket_name)

	files_coll = db[f"{bucket_name}.files"]
	# Find the latest file by uploadDate (no model_name filter)
	doc = files_coll.find_one({}, sort=[("uploadDate", -1)])
	if not doc:
		return None, None

	metadata = doc.get("metadata", {})
	file_id = doc["_id"]
	try:
		raw = fs.get(file_id).read()
		import pickle
		return pickle.loads(raw), metadata
	except Exception:
		return None, metadata


def clear_database(collection_name: str = "features", bucket_name: str = "models", drop_db: bool = False) -> dict:
	"""Clear all engineered features and model registry files from the database.

	This will delete all documents from the specified `collection_name` and
	remove all files stored in the specified GridFS `bucket_name`. If
	`drop_db` is True the entire database will be dropped after removals.

	Returns a dict with counts: {"features_deleted": int, "models_deleted": int}.
	"""
	# Read env and connect (use sensible defaults when env vars are missing)
	mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
	db_name = os.environ.get("DB_NAME", "ml_platform")
	client = MongoClient(mongo_uri)
	db = client[db_name]

	# Clear feature documents
	feature_collection = db[collection_name]
	try:
		res = feature_collection.delete_many({})
		features_deleted = getattr(res, "deleted_count", 0)
	except Exception:
		features_deleted = 0

	# Clear GridFS model files
	import gridfs
	fs = gridfs.GridFS(db, collection=bucket_name)
	files_coll = db[f"{bucket_name}.files"]
	models_deleted = 0
	try:
		existing = list(files_coll.find({}))
		for ef in existing:
			try:
				fs.delete(ef["_id"])
				models_deleted += 1
			except Exception:
				# ignore deletion errors for individual files
				pass
	except Exception:
		models_deleted = 0

	# Optionally drop the entire database
	if drop_db:
		try:
			client.drop_database(db_name)
		except Exception:
			pass

	return {"features_deleted": features_deleted, "models_deleted": models_deleted}

