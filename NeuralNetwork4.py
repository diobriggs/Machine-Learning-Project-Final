import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Load datasets using np.loadtxt
train_data = np.loadtxt('datasets/TrainData4.txt')
train_labels = np.loadtxt('datasets/TrainLabel4.txt')
test_data = np.loadtxt('datasets/TestData4.txt')

# Adjust labels to be in the range [0, 8] (i.e., subtract 1 from the labels)
train_labels = train_labels - 1
# If there are test labels, adjust them similarly
# test_labels = test_labels - 1  # Uncomment if you have test labels

# Replace missing values (1e+99) with NaN
train_data[train_data == 1.00000000000000e+99] = np.nan
test_data[test_data == 1.00000000000000e+99] = np.nan

# Initialize the SimpleImputer to replace missing values with column mean
imputer = SimpleImputer(strategy='mean')

# Fit and transform the training data
train_data_imputed = imputer.fit_transform(train_data)
# Transform the test data using the fitted imputer
test_data_imputed = imputer.transform(test_data)

# Standardize the data (important for neural networks)
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data_imputed)
test_data_scaled = scaler.transform(test_data_imputed)

# Apply PCA for feature selection (adjust the number of components as needed)
pca = PCA(n_components=50)  # You can experiment with different numbers of components
train_data_pca = pca.fit_transform(train_data_scaled)
test_data_pca = pca.transform(test_data_scaled)

# Use Stratified K-Fold Cross-Validation to evaluate the model
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_index, val_index in skf.split(train_data_pca, train_labels):
    X_train_cv, X_val_cv = train_data_pca[train_index], train_data_pca[val_index]
    y_train_cv, y_val_cv = train_labels[train_index], train_labels[val_index]

    # Define the model with L2 regularization on the Dense layers
    model_cv = Sequential()
    model_cv.add(Dense(128, input_dim=X_train_cv.shape[1], activation='relu',
                       kernel_regularizer=tf.keras.regularizers.l2(0.01)))  # L2 regularization
    model_cv.add(Dropout(0.2))  # Dropout layer to avoid overfitting
    model_cv.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))  # L2 regularization
    model_cv.add(Dense(9, activation='softmax'))  # 9 output classes (0-8)
    model_cv.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Manual Early Stopping
    best_val_loss = np.inf
    epochs_no_improve = 0
    patience = 10  # Number of epochs with no improvement after which to stop training
    best_weights = None

    for epoch in range(50):  # Maximum number of epochs
        # Train the model for one epoch
        history = model_cv.fit(X_train_cv, y_train_cv, epochs=1, batch_size=32, verbose=0,
                               validation_data=(X_val_cv, y_val_cv))

        val_loss = history.history['val_loss'][0]

        # Check for improvement in validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model_cv.get_weights()  # Save the best weights
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Stop training if no improvement for 'patience' epochs
        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch + 1} epochs with no improvement.")
            break

    # Restore the best weights
    model_cv.set_weights(best_weights)

    # Evaluate the model
    y_val_pred = model_cv.predict(X_val_cv)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)

    # Compute the accuracy for this fold
    fold_accuracy = accuracy_score(y_val_cv, y_val_pred_classes)
    cv_scores.append(fold_accuracy)

# Print the cross-validation results
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))

# Final model for predicting test data
final_model = Sequential()
final_model.add(Dense(128, input_dim=train_data_pca.shape[1], activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.01)))  # L2 regularization
final_model.add(Dropout(0.2))  # Dropout layer to avoid overfitting
final_model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))  # L2 regularization
final_model.add(Dense(9, activation='softmax'))  # 9 output classes (0-8)
final_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Manual Early Stopping for final model
best_val_loss = np.inf
epochs_no_improve = 0
patience = 10
best_weights = None

for epoch in range(50):
    history = final_model.fit(train_data_pca, train_labels, epochs=1, batch_size=32,
                              verbose=0, validation_data=(train_data_pca, train_labels))

    val_loss = history.history['val_loss'][0]

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = final_model.get_weights()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping after {epoch + 1} epochs with no improvement.")
        break

# Restore the best weights
final_model.set_weights(best_weights)

# Predict on the test dataset
test_predictions = final_model.predict(test_data_pca)
test_pred_classes = np.argmax(test_predictions, axis=1)

test_pred_classes_shifted = test_pred_classes + 1

# Write predictions to file
np.savetxt('results/BriggsClassification4.txt', test_pred_classes_shifted, fmt='%d')

# Evaluate the final model on the full training data
y_pred = final_model.predict(train_data_pca)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print evaluation metrics for the final model
print("Final model Accuracy Score:", accuracy_score(train_labels, y_pred_classes))
print("Final model Classification Report:\n", classification_report(train_labels, y_pred_classes))
print("Final model Confusion Matrix:\n", confusion_matrix(train_labels, y_pred_classes))
