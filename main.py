import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os

# Define file loading and preprocessing functions
def load_data(train_data_file, train_label_file, test_data_file, train_delimiter, test_delimiter):
    # Load train and test data with specified delimiters
    train_data = pd.read_csv(train_data_file, delimiter=train_delimiter, header=None)
    train_labels = pd.read_csv(train_label_file, delimiter=train_delimiter, header=None).values.ravel()
    test_data = pd.read_csv(test_data_file, delimiter=test_delimiter, header=None)

    # Replace missing values (1.0e+99) with NaN, then fill with column means
    train_data.replace(1.0e+99, np.nan, inplace=True)
    train_data.fillna(train_data.mean(), inplace=True)
    test_data.replace(1.0e+99, np.nan, inplace=True)
    test_data.fillna(test_data.mean(), inplace=True)

    return train_data, train_labels, test_data

def train_and_evaluate_model(train_data, train_labels, test_data, n_classes):
    # Normalize the data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Define the ANN model
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=2000, random_state=42)

    # Train the model
    model.fit(train_data, train_labels)

    # Predict and evaluate
    test_predictions = model.predict(test_data)
    return test_predictions

# Main processing loop for each dataset
def process_datasets(results_folder):
    for i in range(1, 6):
        # Construct file paths
        train_data_file = os.path.join(results_folder, f"TrainData{i}.txt")
        train_label_file = os.path.join(results_folder, f"TrainLabel{i}.txt")
        test_data_file = os.path.join(results_folder, f"TestData{i}.txt")

        # Set delimiters for each dataset
        if i == 3:
            train_delimiter = r'\s+'  # Tab or whitespace for training data
            test_delimiter = ','      # Comma for test data
        else:
            train_delimiter = test_delimiter = r'\s+'  # Tab or whitespace for other datasets

        # Load and preprocess data
        train_data, train_labels, test_data = load_data(train_data_file, train_label_file, test_data_file, train_delimiter, test_delimiter)

        # Define number of classes for each dataset
        num_classes = {1: 5, 2: 11, 3: 9, 4: 9, 5: 6}
        n_classes = num_classes[i]

        # Train and evaluate model
        test_predictions = train_and_evaluate_model(train_data, train_labels, test_data, n_classes)

        # Output results for each dataset
        print(f"Dataset {i} - Predicted test labels:\n", test_predictions)

# Set the path to the datasets folder containing the datasets
datasets_folder = 'datasets'
process_datasets(datasets_folder)
