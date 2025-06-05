import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
import os

# Create a simple dataset
data = {
    'Interest_Work_Type': ['Creative', 'Technical', 'Helping', 'Organizing'] * 5,
    'New_Ideas_or_Problem_Solving': ['New Ideas', 'Problem Solving', 'Both Equally'] * 6 + ['New Ideas', 'Problem Solving'],
    'Team_or_Alone': ['Team', 'Alone', 'Both'] * 6 + ['Team', 'Alone'],
    'Boring_Task_Response': ['Find ways to make it interesting', 'Just get it done quickly', 'Procrastinate', 'Ask someone else to do it'] * 5,
    'Logic_Q1': ['Yes', 'No', 'Sometimes'] * 6 + ['Yes', 'No'],
    'Verbal_Q1': ['Apple', 'Banana', 'Carrot', 'Orange'] * 5,
    'Free_Time_Choice': ['Read/Learn', 'Create/Build', 'Socialize', 'Exercise', 'Relax'] * 4,
    'Friend_In_Trouble': ['Offer advice', 'Listen', 'Take action to help', 'Give them space'] * 5,
    'Tech_Liking': ['Yes', 'No', 'Sometimes'] * 6 + ['Yes', 'No'],
    'Career': ['Engineer', 'Artist', 'Teacher', 'Analyst', 'Designer'] * 4
}

df = pd.DataFrame(data)

# Feature Encoding
feature_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X = feature_encoder.fit_transform(df.drop('Career', axis=1))

# Label Encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Career'])

# Model Training
model = SVC(probability=True)
model.fit(X, y)

# Save model and encoders
with open('final_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('feature_encoder.pkl', 'wb') as f:
    pickle.dump(feature_encoder, f)

# Create directory if it doesn't exist
os.makedirs('processed_data', exist_ok=True)
with open('processed_data/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Test model and encoders created successfully!")
