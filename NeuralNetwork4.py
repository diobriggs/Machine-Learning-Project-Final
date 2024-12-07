import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import Input


# Load datasets using np.loadtxt
training_data = np.loadtxt('datasets/TrainData4.txt')
training_labels = np.loadtxt('datasets/TrainLabel4.txt')
testing_data = np.loadtxt('datasets/TestData4.txt')

# Adjust labels to range from 0 to 8
training_labels = training_labels - 1

# Replace missing values with NaN
training_data[training_data == 1.00000000000000e+99] = np.nan
testing_data[testing_data == 1.00000000000000e+99] = np.nan

# Replace NaN values with column mean
imputer = SimpleImputer(strategy='mean')

# Fit and transform the training data
train_data_imputed = imputer.fit_transform(training_data)
# Transform the test data using the fitted imputer
test_data_imputed = imputer.transform(testing_data)

# Standardize the data (important for neural networks)
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data_imputed)
test_data_scaled = scaler.transform(test_data_imputed)

# Cross validation
stratified_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_index, val_index in stratified_kf.split(train_data_scaled, training_labels):
    X_train_cv, X_val_cv = train_data_scaled[train_index], train_data_scaled[val_index]
    y_train_cv, y_val_cv = training_labels[train_index], training_labels[val_index]

    # Define the model with L2 regularization on the Dense layers
    model_cv = Sequential()
    model_cv.add(Input(shape=(X_train_cv.shape[1],)))  # Use Input layer instead of input_dim
    model_cv.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))  # L2 regularization
    model_cv.add(Dropout(0.3))  # Dropout layer to avoid overfitting
    model_cv.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))  # L2 regularization
    model_cv.add(Dense(9, activation='softmax'))  # 9 output classes (0-8)
    model_cv.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Manual Early Stopping
    best_valloss = np.inf
    epochs_wo_improve = 0
    patience = 5
    best_weights = None

    for epoch in range(50):
        history = model_cv.fit(X_train_cv, y_train_cv, epochs=1, batch_size=32, verbose=0,
                               validation_data=(X_val_cv, y_val_cv))

        val_loss = history.history['val_loss'][0]


        if val_loss < best_valloss:
            best_valloss = val_loss
            best_weights = model_cv.get_weights()
            epochs_wo_improve = 0
        else:
            epochs_wo_improve += 1

        # Stop training if no improvement in 5 epochs
        if epochs_wo_improve >= patience:
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
final_model.add(Input(shape=(train_data_scaled.shape[1],)))  # Use Input layer instead of input_dim
final_model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))  # L2 regularization
final_model.add(Dropout(0.3))  # Dropout layer to avoid overfitting
final_model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))  # L2 regularization
final_model.add(Dense(9, activation='softmax'))  # 9 output classes (0-8)
final_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Manual early stopping
best_valloss = np.inf
epochs_wo_improve = 0
patience = 10
best_weights = None

for epoch in range(50):
    history = final_model.fit(train_data_scaled, training_labels, epochs=1, batch_size=32,
                              verbose=0, validation_data=(train_data_scaled, training_labels))

    val_loss = history.history['val_loss'][0]

    if val_loss < best_valloss:
        best_valloss = val_loss
        best_weights = final_model.get_weights()
        epochs_wo_improve = 0
    else:
        epochs_wo_improve += 1

    if epochs_wo_improve >= patience:
        print(f"Early stopping after {epoch + 1} epochs with no improvement.")
        break

# Restore the best weights
final_model.set_weights(best_weights)

# Predict on the test dataset
testing_predictions = final_model.predict(test_data_scaled)
testing_pred_classes = np.argmax(testing_predictions, axis=1)

testing_pred_classes_shifted = testing_pred_classes + 1

# Write predictions to file
np.savetxt('results/BriggsClassification4.txt', testing_pred_classes_shifted, fmt='%d')

# Evaluate the final model on the full training data
y_pred = final_model.predict(train_data_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

# Evaluation metrics
print("Final model Accuracy Score:", accuracy_score(training_labels, y_pred_classes))
print("Final model Classification Report:\n", classification_report(training_labels, y_pred_classes))
print("Final model Confusion Matrix:\n", confusion_matrix(training_labels, y_pred_classes))
