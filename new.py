import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime

def extract_time_features(df):
    dt = pd.to_datetime(df['trans_date'] + ' ' + df['trans_time'], errors='coerce')
    df['trans_hour'] = dt.dt.hour
    df['trans_dayofweek'] = dt.dt.dayofweek
    return df

def lump_categories(df, cat_columns, max_unique=50):
    """Lump rare categories into an 'other' category for each categorical column."""
    for col in cat_columns:
        if col in df.columns and df[col].dtype == 'object':
            # Get the most frequent categories
            top_cats = df[col].value_counts().index[:max_unique]
            df[col] = np.where(df[col].isin(top_cats), df[col], 'other')
    return df

print("Loading training data...")
train = pd.read_csv("train.csv")

required_cols = ['is_fraud', 'trans_date', 'trans_time']
missing_cols = [c for c in required_cols if c not in train.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in training data: {missing_cols}")

y = train['is_fraud']
X = train.drop('is_fraud', axis=1)

print("Extracting time-based features for training...")
X = extract_time_features(X)

# Drop unnecessary columns
cols_to_drop = ['id', 'trans_num', 'trans_date', 'trans_time', 'dob', 'first', 'last', 'street', 'merchant']
X.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')

numeric_features = ['unix_time', 'amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'trans_hour', 'trans_dayofweek']
categorical_features = ['category', 'gender', 'city', 'state', 'job']

numeric_features = [f for f in numeric_features if f in X.columns]
categorical_features = [f for f in categorical_features if f in X.columns]

# Lump categories to avoid huge cardinalities
print("Lumping rare categories to reduce cardinality...")
X = lump_categories(X, categorical_features, max_unique=50)

# Convert to category dtype
for col in categorical_features:
    if col in X.columns:
        X[col] = X[col].astype('category')

print("Creating LightGBM dataset...")
train_data = lgb.Dataset(X, label=y, categorical_feature=categorical_features, free_raw_data=False)

# GPU-friendly parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'device_type': 'gpu',   # enable GPU
    'n_jobs': -1,
    'seed': 42,
    'verbose': -1,
    'max_bin': 63,          # Small bin size for GPU
    'feature_pre_filter': False,  # Allow LightGBM to handle features even if low gain
    'gpu_use_dp': True      # Use double precision if needed
}

print("Starting model training with LightGBM (GPU)...")
num_boost_round = 100
model = lgb.train(params, train_data, num_boost_round=num_boost_round)
print("Training completed successfully.")

print("Loading test data...")
test = pd.read_csv("test.csv")
if 'id' not in test.columns:
    raise ValueError("Test data must have an 'id' column.")

test_ids = test['id']

print("Extracting time-based features from test data...")
test = extract_time_features(test)
test.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')

# Lump categories in test as well, using the same logic
print("Lumping rare categories in test data...")
test = lump_categories(test, categorical_features, max_unique=50)

for col in categorical_features:
    if col in test.columns:
        test[col] = test[col].astype('category')

print("Predicting on test data...")
predictions = model.predict(test)
pred_labels = (predictions >= 0.5).astype(int)

print("Creating submission file...")
submission = pd.DataFrame({'id': test_ids, 'is_fraud': pred_labels})
submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created successfully.")
