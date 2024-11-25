import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load datasets
train_data = np.loadtxt('datasets/TrainData3.txt')
train_labels = np.loadtxt('datasets/TrainLabel3.txt')
test_data = np.loadtxt('datasets/TestData3.txt', delimiter=',')

# Replace missing values (1.00000000000000e+99) with the mean of the feature column
imputer = SimpleImputer(missing_values=1.00000000000000e+99, strategy='mean')
train_data_imputed = imputer.fit_transform(train_data)
test_data_imputed = imputer.transform(test_data)

# Standardize the data
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data_imputed)
test_data_scaled = scaler.transform(test_data_imputed)

# Adjust RandomForestClassifier parameters for stronger regularization
clf = RandomForestClassifier(
    n_estimators=150,          # Slightly increased number of trees
    max_depth=12,              # Limit the depth of the trees for better generalization
    min_samples_split=20,      # Increase the minimum samples per split
    min_samples_leaf=10,        # Increase the minimum samples per leaf
    max_features='sqrt',       # Use a subset of features for each split
    random_state=42,
    bootstrap=True             # Use bootstrapping to improve generalization
)

# Create a pipeline to standardize data and train the model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', clf)
])

# Perform cross-validation to check generalization
cv_scores = cross_val_score(pipeline, train_data_imputed, train_labels, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {np.mean(cv_scores)}")

# Train the RandomForest with the training data using the pipeline
pipeline.fit(train_data_imputed, train_labels)

# Evaluate the model on the training data
train_predictions = pipeline.predict(train_data_imputed)

# Print evaluation metrics
print("Training Accuracy:", accuracy_score(train_labels, train_predictions))
print("\nClassification Report:")
print(classification_report(train_labels, train_predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(train_labels, train_predictions))

# Implement Grid Search to find the best hyperparameters
param_grid = {
    'clf__n_estimators': [100, 150, 200],     # Slightly increase the range of estimators
    'clf__max_depth': [7, 10, 12],            # Keep tree depths shallow to avoid overfitting
    'clf__min_samples_split': [10, 15, 20],   # Test different splits for better regularization
    'clf__min_samples_leaf': [5, 7, 10],      # Test minimum samples per leaf
    'clf__max_features': ['sqrt', 'log2']     # Keep max_features limited
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(train_data_imputed, train_labels)

# Print the best hyperparameters found by Grid Search
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_}")

# Re-train the model using the best parameters found
best_pipeline = grid_search.best_estimator_
best_pipeline.fit(train_data_imputed, train_labels)

# Predict on the training data with the best model
train_predictions_best = best_pipeline.predict(train_data_imputed)

# Print evaluation metrics for the best model
print("\nBest Model Training Accuracy:", accuracy_score(train_labels, train_predictions_best))
print("\nBest Model Classification Report:")
print(classification_report(train_labels, train_predictions_best))
print("\nBest Model Confusion Matrix:")
print(confusion_matrix(train_labels, train_predictions_best))

# Use the best model to predict on the test data
predicted_labels_best = best_pipeline.predict(test_data_imputed)

# Save the predicted labels from the best model to BriggsClassification3.txt
np.savetxt('results/BriggsClassification3.txt', predicted_labels_best, fmt='%d')
