import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import time
import os

# Load preprocessed data
X_train = pd.read_csv("/home/ubuntu/processed_data/X_train.csv")
y_train = pd.read_csv("/home/ubuntu/processed_data/y_train.csv")["target"]
X_test = pd.read_csv("/home/ubuntu/processed_data/X_test.csv")
y_test = pd.read_csv("/home/ubuntu/processed_data/y_test.csv")["target"]

# Load the label encoder to get class names
with open("/home/ubuntu/processed_data/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
class_names = label_encoder.classes_

# Define parameter grids for tuning
param_grids = {
    "Logistic Regression": {
        "C": [0.1, 1, 10],
        "solver": ["liblinear", "saga"],
        "max_iter": [1000, 2000]
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },
    "SVC": {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf", "linear"]
    }
}

models_to_tune = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVC": SVC(random_state=42, probability=True) # Add probability=True for potential future use
}

best_models = {}
tuning_results = {}

print("--- Starting Hyperparameter Tuning --- ")

# Create directory for tuned models
tuned_models_dir = "/home/ubuntu/tuned_models"
os.makedirs(tuned_models_dir, exist_ok=True)

for name, model in models_to_tune.items():
    print(f"\nTuning {name}...")
    start_time = time.time()
    
    grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    tuning_time = time.time() - start_time
    print(f"Tuning completed in {tuning_time:.2f} seconds.")
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best Parameters for {name}: {best_params}")
    print(f"Best CV Accuracy for {name}: {best_score:.4f}")
    
    # Evaluate the best model on the test set
    print(f"Evaluating best {name} on test set...")
    start_time = time.time()
    y_pred = best_model.predict(X_test)
    eval_time = time.time() - start_time
    test_accuracy = accuracy_score(y_test, y_pred)
    test_report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    
    print(f"Test Accuracy for best {name}: {test_accuracy:.4f}")
    # print("Test Classification Report:")
    # print(test_report)
    
    # Save the best model
    model_filename = os.path.join(tuned_models_dir, f"{name.replace(' ', '_').lower()}_best.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Best {name} model saved to {model_filename}")
    
    best_models[name] = best_model
    tuning_results[name] = {
        "best_params": best_params,
        "best_cv_score": best_score,
        "test_accuracy": test_accuracy,
        "test_classification_report": test_report,
        "tuning_time": tuning_time,
        "eval_time": eval_time
    }

# Save tuning results to a file
tuning_summary = "# Hyperparameter Tuning Report\n\n"
for name, metrics in tuning_results.items():
    tuning_summary += f"## {name}\n"
    tuning_summary += f"- Best CV Accuracy: {metrics['best_cv_score']:.4f}\n"
    tuning_summary += f"- Test Accuracy: {metrics['test_accuracy']:.4f}\n"
    tuning_summary += f"- Best Parameters: {metrics['best_params']}\n"
    tuning_summary += f"- Tuning Time: {metrics['tuning_time']:.2f}s\n"
    tuning_summary += f"- Evaluation Time: {metrics['eval_time']:.2f}s\n\n"
    tuning_summary += "### Test Classification Report\n"
    tuning_summary += "```\n"
    tuning_summary += metrics['test_classification_report']
    tuning_summary += "\n```\n\n"

with open("/home/ubuntu/tuning_report.md", "w") as f:
    f.write(tuning_summary)

print("--- Hyperparameter Tuning Complete. Report saved to tuning_report.md ---")

