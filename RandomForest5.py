import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.impute import SimpleImputer

# Loading the data
training_data = np.loadtxt('datasets/TrainData5.txt')
training_labels = np.loadtxt('datasets/TrainLabel5.txt')
testing_data = np.loadtxt('datasets/TestData5.txt')

# Replace missing values with NaN
training_data = np.where(training_data == 1.00000000000000e+99, np.nan, training_data)
testing_data = np.where(testing_data == 1.00000000000000e+99, np.nan, testing_data)

# Fill missing values with the median of each column
imputer = SimpleImputer(strategy='median')
training_data_imputed = imputer.fit_transform(training_data)
testing_data_imputed = imputer.transform(testing_data)


rf_clf = RandomForestClassifier(
    n_estimators=75,
    random_state=42,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced'
)

# Cross validation
stratified_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store metrics for different folds
accuracies = []
precisions = []
recalls = []
f1_scores = []


for training_index, val_index in stratified_kf.split(training_data_imputed, training_labels):
    X_train, X_val = training_data_imputed[training_index], training_data_imputed[val_index]
    y_train, y_val = training_labels[training_index], training_labels[val_index]

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

# Eval Metrics
print(f"Stratified K-Fold Cross-Validation Results:")
print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

# Training accuracy
rf_clf.fit(training_data_imputed, training_labels)
training_data_pred = rf_clf.predict(training_data_imputed)
training_accuracy = accuracy_score(training_labels, training_data_pred)

print(f"Training Accuracy on the entire dataset: {training_accuracy:.4f}")

# Predict on the test set
y_test_pred = rf_clf.predict(testing_data_imputed)


# Save the predicted labels
np.savetxt('results/BriggsClassification5.txt', y_test_pred, fmt='%d')