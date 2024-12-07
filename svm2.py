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

# Loading in the dataset
training_data = np.loadtxt('datasets/TrainData2.txt')
training_labels = np.loadtxt('datasets/TrainLabel2.txt', dtype=int)
testing_data = np.loadtxt('datasets/TestData2.txt')

# Find and replace all missing values in the dataset with NaN
missing_value = 1.0e+99
training_data[training_data == missing_value] = np.nan
testing_data[testing_data == missing_value] = np.nan


# Impute all the NaN values with the mean of each feature.
imputer = SimpleImputer(strategy='mean')
training_data = imputer.fit_transform(training_data)
testing_data = imputer.transform(testing_data)


# Standardize the data
scaler = StandardScaler()
training_data = scaler.fit_transform(training_data)
testing_data = scaler.transform(testing_data)

# Feature Selection
feature_selector = SelectKBest(score_func=f_classif)

pca = PCA(n_components=0.95)

# Cross validation
stratified_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# SVM with balanced class weights
svm = SVC(class_weight='balanced')

variance_filter = VarianceThreshold(threshold=0)


pipeline = Pipeline([
    ('variance_filter', variance_filter),
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('pca', PCA(n_components=0.95)),
    ('svm', SVC(kernel='rbf', class_weight='balanced'))  # Use RBF kernel and balance classes
])


param_grid = {
    'feature_selection__k': [500, 750, 1000],  # Select top k features
    'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  # A range of values for C
    'svm__gamma': [0.0001, 0.001, 0.01, 0.1, 1]  # A range of values for gamma
}


grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_kf, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(training_data, training_labels)

# Best model after tuning
best_model = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)

# Predict on the training data to check performance
training_predictions = best_model.predict(training_data)
print("\nTraining Set Evaluation:")
print("Accuracy:", accuracy_score(training_labels, training_predictions))
print("Classification Report:")
print(classification_report(training_labels, training_predictions))
print("Confusion Matrix:")
print(confusion_matrix(training_labels, training_predictions))

# Cross-val scores
crossv_scores = cross_val_score(best_model, training_data, training_labels, cv=stratified_kf)
print("\nCross-Validation Accuracy Scores:", crossv_scores)
print("Mean Cross-Validation Accuracy:", np.mean(crossv_scores))

# Predict on the test data
test_predictions = best_model.predict(testing_data)

# Save the predictions to the results folder
output_folder = 'results'
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, 'BriggsClassification2.txt')
np.savetxt(output_file, test_predictions, fmt='%d')

print(f"\nPredicted labels saved to {output_file}")
