import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load datasets
training_data = np.loadtxt('datasets/TrainData3.txt')
training_labels = np.loadtxt('datasets/TrainLabel3.txt')
testing_data = np.loadtxt('datasets/TestData3.txt', delimiter=',')

# Find and replace all missing values in the dataset with column mean
imputer = SimpleImputer(missing_values=1.00000000000000e+99, strategy='mean')
training_data_imputed = imputer.fit_transform(training_data)
testing_data_imputed = imputer.transform(testing_data)

# Standardize the data
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(training_data_imputed)
test_data_scaled = scaler.transform(testing_data_imputed)


clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42,
    bootstrap=True
)

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', clf)
])

# Cross validation
cv_scores = cross_val_score(pipeline, training_data_imputed, training_labels, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {np.mean(cv_scores)}")


pipeline.fit(training_data_imputed, training_labels)

# Evaluate the model on the training data
training_predictions = pipeline.predict(training_data_imputed)


param_grid = {
    'clf__n_estimators': [25, 50, 100],
    'clf__max_depth': [5, 7, 10],
    'clf__min_samples_split': [5, 10],
    'clf__min_samples_leaf': [2, 4],
    'clf__max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(training_data_imputed, training_labels)


print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_}")

# Retrain model with best pipeline
best_pipeline = grid_search.best_estimator_
best_pipeline.fit(training_data_imputed, training_labels)

# Predict on the training data with the best model
training_predictions_best = best_pipeline.predict(training_data_imputed)

# Evaluation metrics
print("\nBest Model Training Accuracy:", accuracy_score(training_labels, training_predictions_best))
print("\nBest Model Classification Report:")
print(classification_report(training_labels, training_predictions_best))
print("\nBest Model Confusion Matrix:")
print(confusion_matrix(training_labels, training_predictions_best))


best_predicted_labels = best_pipeline.predict(testing_data_imputed)

# Save the predicted labels from the best model to BriggsClassification3.txt
np.savetxt('results/BriggsClassification3.txt', best_predicted_labels, fmt='%d')
