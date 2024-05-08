import pandas as pd
from imblearn.over_sampling import ADASYN

# Load your dataset
data_path = 'merged_data.csv'  # Make sure to replace this with your actual file path
data = pd.read_csv(data_path)

# Separate your input features and the outcome variable
X = data.drop('outcome', axis=1)
y = data['outcome']

# Initialize the ADASYN model
ada = ADASYN(random_state=42)

# Fit and apply the ADASYN model
X_resampled, y_resampled = ada.fit_resample(X, y)

# Combine the resampled features and outcomes back into a dataframe
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
resampled_data['outcome'] = y_resampled

# Save the balanced dataset to a new CSV file
resampled_data_path = 'resampled_data.csv'
resampled_data.to_csv(resampled_data_path, index=False)

print("The dataset has been balanced and saved to:", resampled_data_path)
