import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle

# Load the dataset
df = pd.read_csv("/home/ubuntu/upload/synthetic_career_test_data.csv")

# Drop the identifier column
df = df.drop("Student_ID", axis=1)

# Separate features (X) and target (y)
X = df.drop("Future_Vision", axis=1)
y = df["Future_Vision"]

# --- Preprocessing --- 

# 1. Encode Categorical Features (X)
# Using OneHotEncoder for nominal features
feature_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = feature_encoder.fit_transform(X)
# Get feature names after encoding
feature_names = feature_encoder.get_feature_names_out(X.columns)
X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)

print("--- Features Encoded ---")
print(f"Shape after OneHotEncoding: {X_encoded_df.shape}")
# print(X_encoded_df.head())

# 2. Encode Target Variable (y)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("--- Target Encoded ---")
# print(f"Encoded labels sample: {y_encoded[:5]}")
# print(f"Original labels sample: {y[:5].tolist()}")
print(f"Classes found by LabelEncoder: {list(label_encoder.classes_)}")

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded_df, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("--- Data Splitting Complete ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# --- Save Processed Data and Encoders --- 

# Create directory for processed data
import os
os.makedirs("/home/ubuntu/processed_data", exist_ok=True)

# Save split data
X_train.to_csv("/home/ubuntu/processed_data/X_train.csv", index=False)
X_test.to_csv("/home/ubuntu/processed_data/X_test.csv", index=False)
pd.DataFrame(y_train, columns=["target"]).to_csv("/home/ubuntu/processed_data/y_train.csv", index=False)
pd.DataFrame(y_test, columns=["target"]).to_csv("/home/ubuntu/processed_data/y_test.csv", index=False)

# Save encoders
with open("/home/ubuntu/processed_data/feature_encoder.pkl", "wb") as f:
    pickle.dump(feature_encoder, f)
with open("/home/ubuntu/processed_data/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("--- Preprocessing Complete. Processed data and encoders saved. ---")

