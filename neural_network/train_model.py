import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import Model  # Ensure this model has the appropriate output layer
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Data loading and preprocessing
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
data_x = scaler.fit_transform(data_x)

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y.values, test_size=0.2, random_state=2001)

# Tensor conversion
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.flatten(), dtype=torch.float32)  # Flatten y_train to match output dimensions
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.flatten(), dtype=torch.float32)  # Flatten y_test for the same reason

# DataLoader setup
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

# Model configuration
n_features = 49
n_nodes = 98

model = Model(n_features, n_nodes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()  # Using BCEWithLogitsLoss

# Training loop
num_of_epochs = 100
model.train()
for epoch in range(num_of_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))  # Targets reshaped to match output
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch} Loss: {loss.item()}')

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predicted_probs = torch.sigmoid(outputs).flatten()  # Apply sigmoid
    predictions = (predicted_probs > 0.5).float()  # Convert to binary predictions
    correct = (predictions == y_test).float().sum()
    accuracy = correct / len(y_test)

print(f'Accuracy: {accuracy:.4f}')

# Compute the confusion matrix
cm = confusion_matrix(y_test.numpy(), predictions.numpy())
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