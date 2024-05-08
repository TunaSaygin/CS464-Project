import pandas as pd

# Load the datasets
data01_path = 'data01.csv'  # Replace with your actual file path
data_imputed_path = 'data_imputed.csv'  # Replace with your actual file path

data01 = pd.read_csv(data01_path)
data_imputed = pd.read_csv(data_imputed_path)

# Merge the 'outcome' column from data01 into data_imputed
# Make sure both data01 and data_imputed have an 'ID' column to use as a key for merging
merged_data = data_imputed.merge(data01[['ID', 'outcome']], on='ID', how='left')

# Save the merged dataset to a new CSV file
merged_data_path = 'merged_data.csv'  # Replace with your desired output file path
merged_data.to_csv(merged_data_path, index=False)

print("Merged dataset saved to:", merged_data_path)
