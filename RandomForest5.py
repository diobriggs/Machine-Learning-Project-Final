import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.impute import SimpleImputer

# Load data
train_data = np.loadtxt('datasets/TrainData5.txt')
train_labels = np.loadtxt('datasets/TrainLabel5.txt')
test_data = np.loadtxt('datasets/TestData5.txt')

# Replace 1.00000000000000e+99 with NaN to identify missing values
train_data = np.where(train_data == 1.00000000000000e+99, np.nan, train_data)
test_data = np.where(test_data == 1.00000000000000e+99, np.nan, test_data)

# Fill missing values with the median of each column
imputer = SimpleImputer(strategy='median')
train_data_imputed = imputer.fit_transform(train_data)
test_data_imputed = imputer.transform(test_data)

# Initialize the Random Forest Classifier with regularization parameters
rf_clf = RandomForestClassifier(
    n_estimators=75,              # Number of trees
    random_state=42,               # For reproducibility
    max_depth=10,                  # Limit tree depth (regularization)
    min_samples_split=10,           # Minimum samples required to split an internal node
    min_samples_leaf=2,            # Minimum samples required at a leaf node
    max_features='sqrt',
    class_weight='balanced',
    oob_score=True
)

# Initialize StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store metrics for each fold
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Stratified K-Fold Cross Validation
for train_index, val_index in skf.split(train_data_imputed, train_labels):
    X_train, X_val = train_data_imputed[train_index], train_data_imputed[val_index]
    y_train, y_val = train_labels[train_index], train_labels[val_index]

    # Fit the model
    rf_clf.fit(X_train, y_train)

    # Predict on the validation set
    y_pred = rf_clf.predict(X_val)

    # Calculate Accuracy
    accuracy = accuracy_score(y_val, y_pred)
    accuracies.append(accuracy)

    # Calculate Precision, Recall, and F1-Score
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted', zero_division=0)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Print the evaluation metrics
print(f"Stratified K-Fold Cross-Validation Results:")
print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

# Fit the model on the entire training data and calculate training accuracy
rf_clf.fit(train_data_imputed, train_labels)
train_data_pred = rf_clf.predict(train_data_imputed)
train_accuracy = accuracy_score(train_labels, train_data_pred)

print(f"Training Accuracy on the entire dataset: {train_accuracy:.4f}")

# Predict on the test set
y_test_pred = rf_clf.predict(test_data_imputed)

# Create results folder if it doesn't exist
os.makedirs('results', exist_ok=True)

# Save the predicted test labels to a file
np.savetxt('results/BriggsClassification5.txt', y_test_pred, fmt='%d')

print("Predicted test labels saved to 'results/BriggsClassification5.txt'")
