import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import time

# Load preprocessed data
X_train = pd.read_csv("/home/ubuntu/processed_data/X_train.csv")
y_train = pd.read_csv("/home/ubuntu/processed_data/y_train.csv")["target"]
X_test = pd.read_csv("/home/ubuntu/processed_data/X_test.csv")
y_test = pd.read_csv("/home/ubuntu/processed_data/y_test.csv")["target"]

# Load the label encoder to get class names
with open("/home/ubuntu/processed_data/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
class_names = label_encoder.classes_

# Define models to train
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVC": SVC(random_state=42)
}

results = {}

print("--- Starting Model Training and Evaluation ---")

for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds.")

    print(f"Evaluating {name}...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.2f} seconds.")

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    results[name] = {
        "accuracy": accuracy,
        "classification_report": report,
        "train_time": train_time,
        "eval_time": eval_time
    }

    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

# Save results to a file
results_summary = "# Model Comparison Report\n\n"
for name, metrics in results.items():
    results_summary += f"## {name}\n"
    results_summary += f"- Accuracy: {metrics['accuracy']:.4f}\n"
    results_summary += f"- Training Time: {metrics['train_time']:.2f}s\n"
    results_summary += f"- Evaluation Time: {metrics['eval_time']:.2f}s\n\n"
    results_summary += "### Classification Report\n"
    results_summary += "```\n"
    results_summary += metrics['classification_report']
    results_summary += "\n```\n\n"

with open("/home/ubuntu/model_comparison_report.md", "w") as f:
    f.write(results_summary)

print("--- Model Comparison Complete. Report saved to model_comparison_report.md ---")

