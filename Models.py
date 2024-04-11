import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import os
import NSGA_vanilla as nsga
print(os.getcwd())
# Load the dataset
# Note: Replace 'your_data.csv' with the actual path to your CSV file
df = pd.read_csv('data01.csv')

# Split the DataFrame into features and target
# Note: Replace 'target_column' with the name of your target column
X = df.drop('outcome', axis=1)
# X = X.drop('ID', axis=1)
# X = X.drop('group', axis=1)
y = df['outcome']

# Split the data into training and testing sets

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
# categorical_cols = X.select_dtypes(include=['object']).columns
# print("Length: " + str(len(categorical_cols)))
# Create imputers for numerical and categorical data
numerical_imputer = SimpleImputer(strategy='mean')
# categorical_imputer = SimpleImputer(strategy='most_frequent')

# Impute missing values in numerical columns
X_numerical = pd.DataFrame(numerical_imputer.fit_transform(X[numerical_cols]), columns=numerical_cols)

# Check if there are any categorical columns before proceeding
# if len(categorical_cols) > 0:
#     # Impute missing values in categorical columns
#     X_categorical = pd.DataFrame(categorical_imputer.fit_transform(X[categorical_cols]), columns=categorical_cols)
#     # Combine the imputed numerical and categorical data
#     X_imputed = pd.concat([X_numerical, X_categorical], axis=1)
# else:
X_imputed = X_numerical  # Only numerical data present
# X_imputed.to_csv('data_imputed.csv', index=False)
# Output the first few rows of the imputed training data and check for any remaining missing values
print(X_imputed.head())
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
print("Remaining missing values:", X_imputed.isnull().sum().sum())
# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
## after impuding I will train a logistic regression classifier and look at its performance
# Initialize the Logistic Regression classifier
# Initialize and train the logistic regression model
logistic_model = LogisticRegression(max_iter=1500)
logistic_model.fit(X_train_scaled, y_train)
y_pred = logistic_model.predict(X_test_scaled)

#performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

#Support Vector Machine
print("\n\nSupport Vector Machines\n\n")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm_linear = svm_model.predict(X_test_scaled)

svm_accuracy_linear = accuracy_score(y_test, y_pred_svm_linear)
print("SVM (linear kernel) Accuracy:", svm_accuracy_linear)

svm_model = SVC(kernel='poly', random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm_poly = svm_model.predict(X_test_scaled)

svm_accuracy_poly = accuracy_score(y_test, y_pred_svm_poly)
print("SVM (polinomial kernel) Accuracy:", svm_accuracy_poly)


"""
# for i in range(1000):
#     arguement = X_imputed.iloc[[i]].copy()  # Use double brackets to keep it as a DataFrame
#     prediction = logistic_model.predict(arguement)
#     if(prediction[0]==1):
#         print(arguement)
#         print(prediction)
# 
arguement = X_imputed.iloc[[47]].copy()  # Use double brackets to keep it as a DataFrame
prediction = logistic_model.predict(arguement)
print(prediction)
test_input = X_imputed.iloc[0:1, :]  # Taking the first row and ensuring it's a DataFrame
# arguement['age'] = 49
# prediction = pipeline.predict(arguement)
# print(prediction)
# Predict with the modified instance
# prediction = pipeline.predict(arguement)
# print(prediction)
# usually accuracy is 86-87%
#lets implement our NSGA-II algorithm  deap
final_population = nsga.create_counterfactuals(X_imputed.iloc[[47]].to_numpy(),X_imputed.to_numpy(),0,logistic_model.predict,250,100)
# After running the algorithm
print(f"Final Population's Fitness:{final_population[0]}")

# for ind in final_population:
#     print(ind)

print("Hall of Fame Individuals:")"""