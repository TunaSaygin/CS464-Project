from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import get_datasets

# Get the dataset
X_train, y_train, X_val, y_val, X_test, y_test = get_datasets.get_datasets("merged_data.csv")

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Validate the classifier
val_predictions = clf.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {val_accuracy}")

# Finally, evaluate the classifier on the test data
test_predictions = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy}")