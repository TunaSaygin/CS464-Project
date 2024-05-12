import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
import NSGA_vanilla as nsga
import matplotlib.pyplot as plt
import shap
import get_datasets as get_ds
# import dice_ml
# from dice_ml.utils import helpers  # helper functions
# print(os.getcwd())
# # Load the dataset
# # Note: Replace 'your_data.csv' with the actual path to your CSV file
# df = pd.read_csv('data01.csv')

# # Split the DataFrame into features and target
# # Note: Replace 'target_column' with the name of your target column
# X = df.drop('outcome', axis=1)
# X = X.drop('ID', axis=1)
# # X = X.drop('group', axis=1)
# y = df['outcome']

# # Split the data into training and testing sets

# # Identify numerical and categorical columns
# numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
# # categorical_cols = X.select_dtypes(include=['object']).columns
# # print("Length: " + str(len(categorical_cols)))
# # Create imputers for numerical and categorical data
# numerical_imputer = SimpleImputer(strategy='mean')
# # categorical_imputer = SimpleImputer(strategy='most_frequent')

# # Impute missing values in numerical columns
# X_numerical = pd.DataFrame(numerical_imputer.fit_transform(X[numerical_cols]), columns=numerical_cols)

# # Check if there are any categorical columns before proceeding
# # if len(categorical_cols) > 0:
# #     # Impute missing values in categorical columns
# #     X_categorical = pd.DataFrame(categorical_imputer.fit_transform(X[categorical_cols]), columns=categorical_cols)
# #     # Combine the imputed numerical and categorical data
# #     X_imputed = pd.concat([X_numerical, X_categorical], axis=1)
# # else:
# X_imputed = X_numerical  # Only numerical data present
# # X_imputed.to_csv('data_imputed.csv', index=False)
# # Output the first few rows of the imputed training data and check for any remaining missing values
# print(X_imputed.head())


# X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
# print("Remaining missing values:", X_imputed.isnull().sum().sum())
# # Initialize the scaler
# adasyn = ADASYN(random_state=42)

# # Resample the dataset
# X_train, y_train = adasyn.fit_resample(X_train, y_train)
X_train,y_train, X_val, y_val, X_test, y_test = get_ds.get_datasets("merged_data.csv")
# print(X_train['outcome'])
# Create an ADASYN instance
scaler = StandardScaler()
# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)
print(f"Length of X_train_scaled: {len(X_train_scaled)}")
## after impuding I will train a logistic regression classifier and look at its performance
# Initialize the Logistic Regression classifier
# Initialize and train the logistic regression model
logistic_model = LogisticRegression(max_iter=1500)
logistic_model.fit(X_train_scaled, y_train)
y_pred_val = logistic_model.predict(X_val_scaled)
y_pred_test = logistic_model.predict(X_test_scaled)
explainer = shap.Explainer(logistic_model, X_train)
shap_values = explainer(X_test_scaled)
shap.summary_plot(shap_values, X_test_scaled)

#performance
accuracy = accuracy_score(y_val, y_pred_val)
print(f"Logistic Regression val Accuracy: {accuracy}")
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Logistic Regression test Accuracy: {accuracy}")

# confusion matrix for logistic reg
conf_matrix = confusion_matrix(y_test, y_pred_test)
print(conf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels= ["Survived", "Died"])
disp.plot()
plt.title("Confusion Matrix for Logistic Regression)")
plt.show()

#Support Vector Machine
print("\n\nSupport Vector Machines\n")
best_accur_svm_l = 0.0
best_c_svm_l = 0
for i in [0.1, 1, 10, 100]:
    svm_model = SVC(kernel='linear', random_state=42, C=i)
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_val_scaled)
    svm_accuracy_linear = accuracy_score(y_val, y_pred_svm)
    print(f"SVM (linear kernel, C={i}) Accuracy: {svm_accuracy_linear}")
    if(svm_accuracy_linear > best_accur_svm_l):
        best_accur_svm_l = svm_accuracy_linear
        best_c_svm_l = i

svm_model = SVC(kernel='linear', random_state=42, C=best_c_svm_l)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm_linear = svm_model.predict(X_test_scaled)
print(f"SVM test (linear kernel, C={best_c_svm_l}) Accuracy: {svm_accuracy_linear}")

best_accur_svm_p = 0.0
best_c_svm_p = 0
for i in [0.1, 1, 10, 100100]:
    svm_model = SVC(kernel='poly', random_state=42, C=i)
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_val_scaled)
    svm_accuracy_poly = accuracy_score(y_val, y_pred_svm)
    print(f"SVM (polynomial kernel, C={i}) Accuracy: {svm_accuracy_poly}")
    if(svm_accuracy_poly > best_accur_svm_p):
        best_accur_svm_p = svm_accuracy_poly
        best_c_svm_p = i
svm_model = SVC(kernel='poly', random_state=42, C=best_c_svm_p)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm_poly = svm_model.predict(X_test_scaled)

svm_accuracy_poly = accuracy_score(y_test, y_pred_svm_poly)
print(f"SVM test (linear kernel, C={best_c_svm_p}) Accuracy: {svm_accuracy_poly}")


# for i in range(1000):
#     arguement = X_imputed.iloc[[i]].copy()  # Use double brackets to keep it as a DataFrame
#     prediction = logistic_model.predict(arguement)
#     if(prediction[0]==1):
#         print(arguement)
#         print(prediction)
# 
# getting the dead values
X_test_filtered = X_test[y_test == 1]
y_test_filtered = y_test[y_test == 1]
arguement = X_test_filtered.iloc[[0]].copy()  # Use double brackets to keep it as a DataFrame
X_test_filtered_array = X_test_filtered.to_numpy()
y_test_filtered_array = y_test_filtered.to_numpy()
print(y_test_filtered_array)
print(X_test_filtered_array[0])
# print(predicted_logistic_model)
# X_test_filtered_array = X_test_one.to_numpy()
# y_test_filtered_array = y_test_one.to_numpy()
print(y_test_filtered)
print(X_test_filtered)
# print(y_test)
# prediction = logistic_model.predict(arguement)
# print(prediction)
# test_input = X_imputed.iloc[0:1, :]  # Taking the first row and ensuring it's a DataFrame
# arguement['age'] = 49
# prediction = pipeline.predict(arguement)
# print(prediction)
# Predict with the modified instance
# prediction = pipeline.predict(arguement)
# print(prediction)
# usually accuracy is 86-87%
#lets implement our NSGA-II algorithm  deap

# for ind in final_population:
#     print(ind)
# X_train_dice = X_train.copy()
# X_train_dice['Outcome'] = y_train
# svm_wrapper = CalibratedClassifierCV(svm_model)
# svm_wrapper.fit(X_train,y_train)
# Prepare data for DiCE
# d =dice_ml.Data(dataframe=X_train_dice, continuous_features=X_train.columns.tolist(), outcome_name='Outcome')
# Create a DiCE model object
# m_linear = dice_ml.Model(model=svm_wrapper, backend='sklearn')

# Create DiCE explainers
# exp_linear = dice_ml.Dice(d, m_linear)

# Generate counterfactuals using the linear kernel SVM model
print("Generating counterfactuals for linear SVM...")
# cf_linear = exp_linear.generate_counterfactuals(X_imputed.iloc[[47]], total_CFs=5, desired_class="opposite")
# cf_linear.visualize_as_dataframe(show_only_changes=True)

lr_final_population = nsga.create_counterfactuals(X_test_filtered_array[0],X_test_filtered_array,0,logistic_model.predict,400,100)
print(scaler.inverse_transform([X_test_filtered_array[0]]))
# After running the algorithm
print(f"Final Population's Fitness:{lr_final_population[0]}")
print(f"Final Population's first value{logistic_model.predict(lr_final_population[0]['features'].reshape(1,-1))}")
print("lr_final population[0] = ",lr_final_population[0])
nsga.plot_features(X_test_filtered_array[0],lr_final_population[0]["features"],y_test_filtered_array[0],lr_final_population[0]["prediction"])

nsga.save_counterfactual_results(X_test_filtered_array,logistic_model.predict,"./lr_counterfactual.csv")

svm_final_population = nsga.create_counterfactuals(X_test_filtered_array[0],X_test_filtered_array,0,svm_model.predict,250,100)
print("SVM_final population[0] = ",svm_final_population[0])
nsga.plot_features(X_test_filtered_array[0],svm_final_population[0]["features"],y_test_filtered_array[0],svm_final_population[0]["prediction"])
nsga.save_counterfactual_results(X_test_filtered_array,svm_model.predict,"./svm_counterfactual.csv")

# confusion matrix for svm linear
conf_matrix = confusion_matrix(y_test, y_pred_svm_linear)
print(conf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels= ["Survived", "Died"])
disp.plot()
plt.title("Confusion Matrix for SVM (linear)")
plt.show()

# confusion matrix for svm poly
conf_matrix = confusion_matrix(y_test, y_pred_svm_poly)
print(conf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels= ["Survived", "Died"])
disp.plot()
plt.title("Confusion Matrix for SVM (polynomial)")
plt.show()