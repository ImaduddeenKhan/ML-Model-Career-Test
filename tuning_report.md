# Hyperparameter Tuning Report

## Logistic Regression
- Best CV Accuracy: 0.0738
- Test Accuracy: 0.0950
- Best Parameters: {'C': 10, 'max_iter': 1000, 'solver': 'liblinear'}
- Tuning Time: 1.65s
- Evaluation Time: 0.00s

### Test Classification Report
```
              precision    recall  f1-score   support

     Analyst       0.20      0.20      0.20        15
      Artist       0.00      0.00      0.00        14
     Athlete       0.08      0.07      0.07        14
     Creator       0.12      0.20      0.15        15
    Designer       0.05      0.07      0.06        15
      Doctor       0.17      0.09      0.12        11
    Engineer       0.00      0.00      0.00        13
Entrepreneur       0.00      0.00      0.00        11
  Journalist       0.15      0.29      0.20        14
      Lawyer       0.10      0.08      0.09        13
      Leader       0.07      0.12      0.09        16
    Musician       0.00      0.00      0.00        12
  Politician       0.17      0.10      0.12        10
   Scientist       0.00      0.00      0.00        13
     Teacher       0.13      0.14      0.14        14

    accuracy                           0.10       200
   macro avg       0.08      0.09      0.08       200
weighted avg       0.08      0.10      0.08       200

```

## Random Forest
- Best CV Accuracy: 0.0875
- Test Accuracy: 0.0500
- Best Parameters: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}
- Tuning Time: 3.25s
- Evaluation Time: 0.02s

### Test Classification Report
```
              precision    recall  f1-score   support

     Analyst       0.00      0.00      0.00        15
      Artist       0.06      0.07      0.06        14
     Athlete       0.00      0.00      0.00        14
     Creator       0.06      0.07      0.06        15
    Designer       0.05      0.07      0.06        15
      Doctor       0.00      0.00      0.00        11
    Engineer       0.00      0.00      0.00        13
Entrepreneur       0.14      0.09      0.11        11
  Journalist       0.08      0.07      0.07        14
      Lawyer       0.07      0.08      0.07        13
      Leader       0.04      0.06      0.05        16
    Musician       0.00      0.00      0.00        12
  Politician       0.00      0.00      0.00        10
   Scientist       0.10      0.08      0.09        13
     Teacher       0.12      0.14      0.13        14

    accuracy                           0.05       200
   macro avg       0.05      0.05      0.05       200
weighted avg       0.05      0.05      0.05       200

```

## SVC
- Best CV Accuracy: 0.0950
- Test Accuracy: 0.0950
- Best Parameters: {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}
- Tuning Time: 4.01s
- Evaluation Time: 0.01s

### Test Classification Report
```
              precision    recall  f1-score   support

     Analyst       0.09      0.13      0.11        15
      Artist       0.00      0.00      0.00        14
     Athlete       0.08      0.14      0.10        14
     Creator       0.13      0.27      0.17        15
    Designer       0.08      0.27      0.12        15
      Doctor       0.00      0.00      0.00        11
    Engineer       0.00      0.00      0.00        13
Entrepreneur       0.00      0.00      0.00        11
  Journalist       0.00      0.00      0.00        14
      Lawyer       0.00      0.00      0.00        13
      Leader       0.12      0.44      0.18        16
    Musician       0.00      0.00      0.00        12
  Politician       0.00      0.00      0.00        10
   Scientist       0.00      0.00      0.00        13
     Teacher       0.00      0.00      0.00        14

    accuracy                           0.10       200
   macro avg       0.03      0.08      0.05       200
weighted avg       0.04      0.10      0.05       200

```

