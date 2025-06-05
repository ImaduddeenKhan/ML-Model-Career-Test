import pandas as pd

# Load the dataset
df = pd.read_csv('/home/ubuntu/upload/synthetic_career_test_data.csv')

# Initial exploration
print('--- Dataset Head ---')
print(df.head())
print('\n--- Dataset Info ---')
df.info()
print('\n--- Dataset Description ---')
print(df.describe(include='all'))
print('\n--- Missing Values ---')
print(df.isnull().sum())

# Identify potential features (questions) and target (career)
# Assuming the last column is the target variable 'Suggested Career Stream'
features = df.columns[:-1]
target = df.columns[-1]

print(f'\nFeatures: {list(features)}')
print(f'Target: {target}')

# Analyze the target variable distribution
print('\n--- Target Variable Distribution ---')
print(df[target].value_counts())

# Basic Preprocessing Steps (Example - will depend on actual data)
# Check data types - if questions are categorical, they might need encoding
# For now, let's assume questions are numerical or ordinal based on the describe output
# If categorical encoding is needed, we can use OneHotEncoder or LabelEncoder later

# Save the analysis summary
analysis_summary = f"""
Dataset Shape: {df.shape}

Columns: {list(df.columns)}

Data Types:
{df.dtypes}

Missing Values Summary:
{df.isnull().sum()}

Target Variable ('{target}') Distribution:
{df[target].value_counts()}
"""

# Note: No actual preprocessing like encoding or scaling is done yet.
# This script focuses purely on exploration.

print("\n--- Data Exploration Complete ---")

