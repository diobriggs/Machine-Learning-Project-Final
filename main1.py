import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

# Load dataset from text files
def load_data(train_data_file, train_label_file, test_data_file):
    # Load data from files with tab delimiter
    train_data = np.loadtxt(train_data_file, delimiter=None)
    train_labels = np.loadtxt(train_label_file, delimiter=None)
    test_data = np.loadtxt(test_data_file, delimiter=None)

    # Replace missing values (1.00000000000000e+99) with NaN for easier handling
    train_data[train_data == 1.00000000000000e+99] = np.nan
    test_data[test_data == 1.00000000000000e+99] = np.nan

    # Fill missing values with column mean for training and testing data
    col_mean_train = np.nanmean(train_data, axis=0)
    inds_train = np.where(np.isnan(train_data))
    train_data[inds_train] = np.take(col_mean_train, inds_train[1])

    col_mean_test = np.nanmean(test_data, axis=0)
    inds_test = np.where(np.isnan(test_data))
    test_data[inds_test] = np.take(col_mean_test, inds_test[1])

    return train_data, train_labels, test_data

# Neural Network Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, 512)   # First hidden layer
        self.fc2 = nn.Linear(512, 256)          # Second hidden layer
        self.fc3 = nn.Linear(256, num_classes)  # Output layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)        # Dropout to prevent overfitting

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Training the model
def train_model(train_data, train_labels, input_size, num_classes, epochs=100, batch_size=32):
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        train_data, train_labels, test_size=0.2, random_state=42
    )

    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Initialize model, loss, and optimizer
    model = NeuralNetwork(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    best_val_accuracy = 0
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_accuracy = accuracy_score(y_val, val_predictions)

        # Early stopping check
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Load the best model state
    model.load_state_dict(best_model_state)
    return model, scaler

# Evaluation on the test dataset
def evaluate_model(model, scaler, test_data):
    # Standardize test data using the training set's scaler
    test_data = scaler.transform(test_data)
    X_test = torch.tensor(test_data, dtype=torch.float32)

    # Model predictions
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_predictions = torch.argmax(test_outputs, dim=1)

    # Convert predictions to 1-based
    test_predictions = test_predictions + 1  # Shift to 1-based class labels

    return test_predictions.numpy()

# Main function to run the script
def main():
    # Loop through datasets 1 and 2
    for dataset_num in [1,2,4]:  # Stops at dataset 2
        # File paths
        train_data_file = f'datasets/TrainData{dataset_num}.txt'
        train_label_file = f'datasets/TrainLabel{dataset_num}.txt'
        test_data_file = f'datasets/TestData{dataset_num}.txt'

        # Load data
        train_data, train_labels, test_data = load_data(
            train_data_file, train_label_file, test_data_file
        )

        # Adjust labels to 0-based indexing for model training
        train_labels -= 1

        # Determine input size and number of classes
        input_size = train_data.shape[1]
        num_classes = len(np.unique(train_labels))

        # Train the model
        model, scaler = train_model(train_data, train_labels, input_size, num_classes)

        # Evaluate the model
        test_predictions = evaluate_model(model, scaler, test_data)
        print(f"Test Predictions for Dataset {dataset_num}:", test_predictions)

if __name__ == "__main__":
    main()