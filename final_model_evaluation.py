import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np

# Load test data
X_test = pd.read_csv("/home/ubuntu/processed_data/X_test.csv")
y_test = pd.read_csv("/home/ubuntu/processed_data/y_test.csv")["target"]

# Load the label encoder to get class names
with open("/home/ubuntu/processed_data/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
class_names = label_encoder.classes_

# Load the best models
tuned_models_dir = "/home/ubuntu/tuned_models"
model_files = {
    "Logistic Regression": os.path.join(tuned_models_dir, "logistic_regression_best.pkl"),
    "Random Forest": os.path.join(tuned_models_dir, "random_forest_best.pkl"),
    "SVC": os.path.join(tuned_models_dir, "svc_best.pkl")
}

models = {}
for name, file_path in model_files.items():
    with open(file_path, "rb") as f:
        models[name] = pickle.load(f)

# Create directory for evaluation results
eval_dir = "/home/ubuntu/evaluation_results"
os.makedirs(eval_dir, exist_ok=True)

# Evaluate each model and generate visualizations
results = {}
for name, model in models.items():
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Store results
    results[name] = {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "predictions": y_pred
    }
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f"{name.replace(' ', '_').lower()}_confusion_matrix.png"))
    plt.close()
    
    # Plot precision, recall, f1-score for each class
    metrics_df = pd.DataFrame(report).T
    metrics_df = metrics_df.drop(['accuracy', 'macro avg', 'weighted avg'])
    
    plt.figure(figsize=(14, 8))
    metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar')
    plt.title(f"Classification Metrics by Class - {name}")
    plt.ylabel("Score")
    plt.xlabel("Class")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f"{name.replace(' ', '_').lower()}_class_metrics.png"))
    plt.close()

# Compare model accuracies
accuracies = {name: results[name]["accuracy"] for name in models.keys()}
plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0, 0.2)  # Adjust based on your actual accuracies
plt.tight_layout()
plt.savefig(os.path.join(eval_dir, "model_accuracy_comparison.png"))
plt.close()

# Determine the best model based on accuracy
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
best_accuracy = accuracies[best_model_name]

# Save the best model as the final model
with open("/home/ubuntu/final_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save the feature encoder for future use
with open("/home/ubuntu/processed_data/feature_encoder.pkl", "rb") as f:
    feature_encoder = pickle.load(f)
with open("/home/ubuntu/feature_encoder.pkl", "wb") as f:
    pickle.dump(feature_encoder, f)

# Generate final evaluation report
report_content = f"""# Final Model Evaluation Report

## Model Selection
After training and tuning multiple models, the **{best_model_name}** was selected as the best performing model with a test accuracy of **{best_accuracy:.4f}**.

## Model Comparison
| Model | Accuracy |
|-------|----------|
"""

for name, acc in accuracies.items():
    report_content += f"| {name} | {acc:.4f} |\n"

report_content += """
## Performance Analysis
The overall accuracy of the models is relatively low, which suggests:
1. The dataset might not have strong predictive features for the target variable
2. The career prediction task is complex and might require more features
3. The current feature set might need additional engineering or transformation
4. A larger dataset might be beneficial for better generalization

## Recommendations for Improvement
1. **Feature Engineering**: Create additional features from the existing questions
2. **Data Collection**: Gather more training data with additional questions
3. **Model Exploration**: Try more complex models like neural networks or ensemble methods
4. **Adaptive Questioning**: Implement a system that asks follow-up questions based on previous answers

## Implementation Plan
The selected model will be integrated into a Flask API with the following components:
1. Endpoint for receiving user responses
2. Integration with Gemini API for adaptive questioning
3. PDF report generation with career recommendations
4. Interactive chat-style UI for user engagement

## Visualizations
Please refer to the generated visualization files in the evaluation_results directory for detailed performance metrics.
"""

with open(os.path.join(eval_dir, "final_evaluation_report.md"), "w") as f:
    f.write(report_content)

print(f"Final model ({best_model_name}) saved to /home/ubuntu/final_model.pkl")
print(f"Feature encoder saved to /home/ubuntu/feature_encoder.pkl")
print(f"Evaluation report saved to {os.path.join(eval_dir, 'final_evaluation_report.md')}")
print(f"Visualizations saved to {eval_dir}")
