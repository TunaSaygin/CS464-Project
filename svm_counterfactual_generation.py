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
X_train,y_train, X_val, y_val, X_test, y_test = get_ds.get_datasets("merged_data.csv")
# print(X_train['outcome'])
# Create an ADASYN instance
scaler = StandardScaler()
# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)


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

X_test_filtered = X_test[y_test == 1]
y_test_filtered = y_test[y_test == 1]
X_test_filtered_array = X_test_filtered.to_numpy()
y_test_filtered_array = y_test_filtered.to_numpy()
svm_final_population = nsga.create_counterfactuals(X_test_filtered_array[0],X_test_filtered_array,0,svm_model.predict,250,300)
print("SVM_final population[0] = ",svm_final_population[0])
nsga.plot_features(X_test_filtered_array[0],svm_final_population[0]["features"],y_test_filtered_array[0],svm_final_population[0]["prediction"])
nsga.save_counterfactual_results(X_test_filtered_array,svm_model.predict,"../svm_counterfactual.csv")