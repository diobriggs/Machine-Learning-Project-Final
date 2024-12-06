import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
import os

# Load the dataset
train_data = np.loadtxt('datasets/TrainData2.txt')
train_labels = np.loadtxt('datasets/TrainLabel2.txt', dtype=int)
test_data = np.loadtxt('datasets/TestData2.txt')

# Replace the missing values (1.00000000000000e+99) with NaN for easier processing
missing_value = 1.0e+99
train_data[train_data == missing_value] = np.nan
test_data[test_data == missing_value] = np.nan


# Impute missing values with the mean of each feature
imputer = SimpleImputer(strategy='mean')
train_data = imputer.fit_transform(train_data)
test_data = imputer.transform(test_data)


# Standardize the data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Feature selection using SelectKBest
feature_selector = SelectKBest(score_func=f_classif)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=0.95)  # Retain 80% of the variance

# Use Stratified K-Fold Cross-Validation to evaluate the model
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# SVM with balanced class weights to handle imbalance
svm = SVC(class_weight='balanced')

variance_filter = VarianceThreshold(threshold=0)

# Set up the SVM model pipeline with hyperparameter tuning using GridSearchCV
pipeline = Pipeline([
    ('variance_filter', variance_filter),
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('pca', PCA(n_components=0.95)),
    ('svm', SVC(kernel='rbf', class_weight='balanced'))  # Use RBF kernel and balance classes
])

# Hyperparameters to tune - expanded grid for regularization and kernel parameters
param_grid = {
    'feature_selection__k': [500, 750, 1000],  # Select top k features
    'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  # A range of values for C
    'svm__gamma': [0.0001, 0.001, 0.01, 0.1, 1]  # A range of values for gamma
}

# GridSearchCV for hyperparameter tuning with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(train_data, train_labels)

# Best model after tuning
best_model = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)

# Predict on the training data to check performance
train_predictions = best_model.predict(train_data)
print("\nTraining Set Evaluation:")
print("Accuracy:", accuracy_score(train_labels, train_predictions))
print("Classification Report:")
print(classification_report(train_labels, train_predictions))
print("Confusion Matrix:")
print(confusion_matrix(train_labels, train_predictions))

# Evaluate using cross-validation accuracy score
cv_scores = cross_val_score(best_model, train_data, train_labels, cv=skf)
print("\nCross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", np.mean(cv_scores))

# Predict on the test data
test_predictions = best_model.predict(test_data)

# Save the predictions to the results folder
output_folder = 'results'
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, 'BriggsClassification2.txt')
np.savetxt(output_file, test_predictions, fmt='%d')

print(f"\nPredicted labels saved to {output_file}")
