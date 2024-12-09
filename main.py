import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime


print("Loading data. . .")
train = pd.read_csv("train.csv")

# Separate target
y = train['is_fraud']
X = train.drop('is_fraud', axis=1)

def extract_time_features(df):
    # Combine date and time into a single datetime for easier feature extraction
    dt = pd.to_datetime(df['trans_date'] + ' ' + df['trans_time'])
    df['trans_hour'] = dt.dt.hour
    df['trans_dayofweek'] = dt.dt.dayofweek
    return df

X = extract_time_features(X)

# Drop columns that are not likely useful or are too unique (IDs, names, etc.)
cols_to_drop = ['id', 'trans_num', 'trans_date', 'trans_time', 'dob', 'first', 'last', 'street', 'merchant']
X = X.drop(cols_to_drop, axis=1, errors='ignore')

# Identify categorical and numerical features
numeric_features = ['unix_time', 'amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'trans_hour', 'trans_dayofweek']
categorical_features = ['category', 'gender', 'city', 'state', 'job']
numeric_features = [f for f in numeric_features if f in X.columns]
categorical_features = [f for f in categorical_features if f in X.columns]

print("Building preprocessing Pipelines. . .")

# Preprocessing pipeline for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ],
    remainder='drop'  # Drop any columns not specified
)


print("Building model. . .")
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, verbose=True))])

print("Training model. . .")
model.fit(X, y)

# --------------------------
# Load and Preprocess Test Data
# --------------------------
test = pd.read_csv("test.csv")
test_ids = test['id'].copy()

test = extract_time_features(test)
test = test.drop(cols_to_drop, axis=1, errors='ignore')

# Ensure same columns exist in test; if not, they'll be handled by the pipeline (missing categories etc.)
# No need to drop or reselect columns here as ColumnTransformer will handle present and missing columns gracefully.

# --------------------------
# Predict on Test Data
# --------------------------
predictions = model.predict(test)

# --------------------------
# Create Submission
# --------------------------
submission = pd.DataFrame({'id': test_ids, 'is_fraud': predictions})
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
