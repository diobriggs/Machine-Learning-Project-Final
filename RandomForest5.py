import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE  # Import SMOTE
import os

# Load data
train_data = np.loadtxt('datasets/TrainData5.txt')
train_labels = np.loadtxt('datasets/TrainLabel5.txt')
test_data = np.loadtxt('datasets/TestData5.txt')

# Replace the missing values (1.00000000000000e+99) with NaN for easier processing
missing_value = 1.0e+99
train_data[train_data == missing_value] = np.nan
test_data[test_data == missing_value] = np.nan

# Impute missing values with the mean of each feature
imputer = SimpleImputer(strategy='mean')
train_data = imputer.fit_transform(train_data)
test_data = imputer.transform(test_data)

# Use SMOTE to balance the training dataset
smote = SMOTE(random_state=42)
train_data_smote, train_labels_smote = smote.fit_resample(train_data, train_labels)

# Initialize Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# Perform 5-fold cross-validation to evaluate the model
cv_scores = cross_val_score(clf, train_data_smote, train_labels_smote, cv=5, scoring='accuracy')
print(f"Cross-validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Train the classifier on the full training dataset (after SMOTE)
clf.fit(train_data_smote, train_labels_smote)

# Predict the test data
test_predictions = clf.predict(test_data)

# Save the predicted test labels to a text file
os.makedirs('results', exist_ok=True)
np.savetxt('results/BriggsClassification5.txt', test_predictions, fmt='%d')

# Evaluate model performance using additional metrics
# Use train-test split for detailed evaluation (optional)
X_train, X_valid, y_train, y_valid = train_test_split(train_data_smote, train_labels_smote, test_size=0.2, random_state=42)

# Fit the model on the training split
clf.fit(X_train, y_train)

# Predictions on validation set
y_pred = clf.predict(X_valid)

# Compute evaluation metrics
accuracy = accuracy_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred, average='weighted')
recall = recall_score(y_valid, y_pred, average='weighted')
f1 = f1_score(y_valid, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_valid, y_pred)
class_report = classification_report(y_valid, y_pred)

# Print evaluation metrics
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
