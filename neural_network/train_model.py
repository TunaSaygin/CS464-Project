import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import Model  # Ensure this model has the appropriate output layer
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import sys

path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
sys.path.insert(1, parent_path)
import get_datasets

"""# Data loading and preprocessing
current_path = os.getcwd()
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))

file_path_x = f'{parent_path}/resampled_data.csv'
file_path_y = f'{parent_path}/resampled_data.csv'

data_x = pd.read_csv(file_path_x)
data_y = pd.read_csv(file_path_y)

data_x = data_x.drop(columns=['ID', 'group'])
data_y = data_y[["outcome"]]

# Standardize features
scaler = StandardScaler()
data_x = scaler.fit_transform(data_x)"""

# Model configuration
n_features = 50
n_nodes = 100
num_of_epochs = 100
train_batch = 64
test_batch = 64

# Get the dataset
X_train, y_train, X_val, y_val, X_test, y_test = get_datasets.get_datasets(parent_path+"/merged_data.csv")

# Tensor conversion
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)  # Flatten y_train to match output dimensions
X_val = torch.tensor(X_test.values, dtype=torch.float32)
y_val = torch.tensor(y_test.values, dtype=torch.float32)  # Flatten y_test for the same reason
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)  # Flatten y_test for the same reason

print(X_train.shape)
print(y_train.shape)

# DataLoader setup
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=train_batch, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=test_batch, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=test_batch, shuffle=False)

# Model, optimizer and loss function
model = Model(n_features, n_nodes)
optimizer = torch.optim.Adam(model.parameters(), lr=6e-3)
criterion = torch.nn.BCEWithLogitsLoss()  # Using BCEWithLogitsLoss

# start training
# for each epoch calculate validation performance
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_acc = 0

# Training loop
for epoch in range(num_of_epochs):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))  # Targets reshaped to match output
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predicted = torch.argmax(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    print("loss: ", running_loss)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    train_acc = 100 * correct / total
    train_accuracies.append(train_acc)
    
    # Validation
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            running_loss += loss.item()

            predicted = torch.argmax(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)

    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)

    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

# Test set
y_pred = []
y_true = []
running_loss = 0.0

model.eval()
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predicted = torch.argmax(outputs.data, 1)
        loss = criterion(outputs, targets.unsqueeze(1))
        running_loss += loss.item()

        y_pred.extend(predicted.numpy())
        y_true.extend(targets.numpy())

losses = running_loss / len(test_loader.dataset)

# Compute accuracy
print()
acc = accuracy_score(y_true, y_pred)
print("Test Accuracy: ", acc)
# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['Survived', 'Died'], yticklabels=['Survived', 'Died'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

"""
# plot the graphs
x = torch.linspace(-5, 5, 100).reshape(-1, 1)
x = torch.hstack(n_features*[x])

for i in range(n_features):
    plt.plot(
        x[:, 0].detach().numpy(),
        model.get_submodule('lr').weight[0][i].item() * model.get_submodule('features')(x)[:, i].detach().numpy())
    plt.title(f'Feature {i+1}')
    plt.show()
"""