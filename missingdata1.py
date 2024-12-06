import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # Enable IterativeImputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Paths
input_folder = "./datasets"
output_folder = "./results"

# Function to load and process datasets
def load_dataset(file_path):
    data = pd.read_csv(file_path, sep="\t", header=None)  # Adjust delimiter if necessary
    # Replace missing value placeholder with NaN for processing
    data.replace(1.0e+99, np.nan, inplace=True)
    return data

# Function to save completed dataset
def save_dataset(data, file_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    data.to_csv(os.path.join(output_folder, file_name), sep="\t", header=False, index=False, float_format='%.15f')

# Imputation function
def impute_missing_values(data, method="iterative", neighbors=5):
    if method == "knn":
        imputer = KNNImputer(n_neighbors=neighbors, weights="distance")
    elif method == "iterative":
        imputer = IterativeImputer(max_iter=200, random_state=42)
    else:
        raise ValueError("Invalid imputation method specified.")

    imputed_data = imputer.fit_transform(data)
    return pd.DataFrame(imputed_data)

# Cross-Validation Function
def cross_validate_imputation(data, method="iterative", neighbors=5, k=5):
    """
    Perform k-fold cross-validation to validate imputation accuracy.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = []

    for train_index, test_index in kf.split(data):
        # Split the data into training and testing sets
        train_data = data.iloc[train_index].copy()
        test_data = data.iloc[test_index].copy()

        # Simulate missing values in the test set (mask 10% of the data)
        test_mask = np.random.choice([True, False], size=test_data.shape, p=[0.1, 0.9])
        test_data_masked = test_data.copy()
        test_data_masked[test_mask] = np.nan

        # Impute the missing values in the masked test set
        imputed_test_data = impute_missing_values(test_data_masked, method=method, neighbors=neighbors)

        # Calculate MSE for the simulated missing values
        true_values = test_data.values[test_mask]  # Get true values where missing
        imputed_values = imputed_test_data.values[test_mask]  # Get imputed values where missing

        # Ensure no NaN values are in the true or imputed values
        valid_mask = ~np.isnan(true_values) & ~np.isnan(imputed_values)
        true_values = true_values[valid_mask]
        imputed_values = imputed_values[valid_mask]

        # Only calculate MSE if both true and imputed values are valid
        if len(true_values) > 0:
            mse = mean_squared_error(true_values, imputed_values)
            mse_scores.append(mse)

    return np.mean(mse_scores)

# Main processing function
def process_datasets():
    datasets = [("MissingData1.txt", "BriggsMissingResult1.txt"),
                ("MissingData2.txt", "BriggsMissingResult2.txt")]

    mse_scores = []
    for input_file, output_file in datasets:
        print(f"Processing {input_file}...")
        # Load dataset
        data = load_dataset(os.path.join(input_folder, input_file))

        # Perform cross-validation for validation
        mse = cross_validate_imputation(data, method="iterative", k=5)
        mse_scores.append(mse)
        print(f"Cross-Validation MSE: {mse:.4f}")

        # Perform final imputation on the entire dataset
        imputed_data = impute_missing_values(data, method="iterative")

        # Save completed dataset
        save_dataset(imputed_data, output_file)
        print(f"{output_file} saved.")

    print("\nCross-Validation MSE Scores:", mse_scores)


process_datasets()
