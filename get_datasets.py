import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN

def get_datasets():
    # Load your dataset
    data_path = 'merged_data.csv'  # Replace with your actual file path
    data = pd.read_csv(data_path)

    # Separate features and target variable
    X = data.drop('group', axis=1)
    X = data.drop('ID', axis=1)
    X = data.drop('outcome', axis=1)  # Assuming 'outcome' is your target variable
    y = data['outcome']

    # First, split the data into training and temporary dataset (70% training, 30% temporary)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Split the temporary dataset into validation and test sets (10% of original data for validation, 20% for test)
    # Note: Validation is 1/3 of 30% (~10%), and Test is 2/3 of 30% (~20%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, stratify=y_temp, random_state=42)

    # Verify the splits
    print("Training set:")
    print(y_train.value_counts)
    print("Validation set:")
    print(y_val.value_counts)
    print("Test set:")
    print(y_test.value_counts)

    # Initialize the ADASYN model
    ada = ADASYN(random_state=42)

    # Fit and apply the ADASYN model
    X_resampled, y_resampled = ada.fit_resample(X_train, y_train)

    # Combine the resampled features and outcomes back into a dataframe
    resampled_data = pd.DataFrame(X_resampled, columns=X_train.columns)
    resampled_data['outcome'] = y_resampled

    print("Train set after generating new data:")
    print(y_resampled.value_counts)

    return X_resampled, y_resampled, X_val, y_val, X_test, y_test
