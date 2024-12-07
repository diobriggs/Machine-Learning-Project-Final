import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
import os

# Loading in the dataset
training_data = np.loadtxt('datasets/TrainData1.txt')
training_labels = np.loadtxt('datasets/TrainLabel1.txt', dtype=int)
testing_data = np.loadtxt('datasets/TestData1.txt')

# Find and replace all missing values in the dataset with NaN
missing_value = 1.0e+99
training_data[training_data == missing_value] = np.nan
testing_data[testing_data == missing_value] = np.nan

# Impute all the NaN values with the mean of each feature.
imputer = SimpleImputer(strategy='mean')
train_data = imputer.fit_transform(training_data)
testing_data = imputer.transform(testing_data)

# Standardize the data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
testing_data = scaler.transform(testing_data)

# Apply PCA while keeping 95% of variance
pca = PCA(n_components=0.95)
train_data_pca = pca.fit_transform(train_data)
test_data_pca = pca.transform(testing_data)

# Cross validation
stratified_kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# SVM with balanced class weights.
svm = SVC(class_weight='balanced')

# Set up the SVM model pipeline with hyperparameter tuning using GridSearchCV
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('svm', SVC(kernel='rbf', class_weight='balanced'))  # Use RBF kernel and balance classes
])

# Hyperparameters to tune - expanded grid for regularization and kernel parameters
param_grid = {
    'svm__C': [0.01, 0.1, 1, 10, 100],
    'svm__gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]
}

# GridSearchCV for hyperparameter tuning with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_kf, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(train_data, training_labels)

# Best model after tuning
best_model = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)

# Predict on the training data to check performance
train_predictions = best_model.predict(train_data)
print("\nTraining Set Evaluation:")
print("Accuracy:", accuracy_score(training_labels, train_predictions))
print("Classification Report:")
print(classification_report(training_labels, train_predictions))
print("Confusion Matrix:")
print(confusion_matrix(training_labels, train_predictions))

# Evaluate using cross-validation accuracy score
cv_scores = cross_val_score(best_model, train_data, training_labels, cv=stratified_kf)
print("\nCross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", np.mean(cv_scores))

# Predict on the test data
test_predictions = best_model.predict(testing_data)

# Save the predictions to the results folder
output_folder = 'results'
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, 'BriggsClassification1.txt')
np.savetxt(output_file, test_predictions, fmt='%d')

print(f"\nPredicted labels saved to {output_file}")
