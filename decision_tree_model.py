from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import get_datasets
import NSGA_vanilla as nsga
import matplotlib.pyplot as plt

# Get the dataset
X_train, y_train, X_val, y_val, X_test, y_test = get_datasets.get_datasets("merged_data.csv")
scaler = StandardScaler()
# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)
# Initialize the Decision Tree Classifier
c_alphas = [0,0.005,0.01,0.015,0.02,0.025,0.030,0.035]
best_c_alpha = 0
best_val_accuracy = 0
for c_alpha in c_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=c_alpha)
    # Train the classifier on the training data
    clf.fit(X_train_scaled, y_train)

    # Validate the classifier
    val_predictions = clf.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"Ccp_alpha = {c_alpha}, validation accuracy: {val_accuracy}")
    if val_accuracy>best_val_accuracy:
        best_c_alpha = c_alpha
print(f"best ccp_alpha: {c_alpha}")
clf = DecisionTreeClassifier(ccp_alpha=best_c_alpha)
clf.fit(X_train_scaled, y_train)
# Finally, evaluate the classifier on the test data
test_predictions = clf.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy}")

# confusion matrix for decision tree
conf_matrix = confusion_matrix(y_test, test_predictions)
print(conf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels= ["Survived", "Died"])
disp.plot()
plt.title("Confusion Matrix for Decision Tree (ccp_alpha=0.035)")
plt.show()

#counterfactuals
X_test_filtered = X_test_scaled[y_test == 1]
y_test_filtered = y_test[y_test == 1]
# X_test_filtered_array = X_test_filtered.to_numpy()
y_test_filtered_array = y_test_filtered.to_numpy()
X_test_filtered_array = X_test_filtered
dtree_final_population = nsga.create_counterfactuals(X_test_filtered_array[0],X_test_filtered_array,0,clf.predict,50,500)
print("dtree_final population[0] = ",dtree_final_population[0])
nsga.plot_features(X_test_filtered_array[0],dtree_final_population[0]["features"],y_test_filtered_array[0],dtree_final_population[0]["prediction"])
nsga.save_counterfactual_results(X_test_filtered_array,clf.predict,"./dtree_counterfactual.csv")