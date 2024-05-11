import torch
import torch.nn as nn
import get_datasets
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(48, 96)
        self.layer_norm1 = nn.LayerNorm(96)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(96, 48)
        self.layer_norm2 = nn.LayerNorm(48)
        self.output_layer = nn.Linear(48, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer_norm1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.layer_norm2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

# Model configuration
num_of_epochs = 100
train_batch = 16
test_batch = 16
learning_rate = 1e-1

# Get the dataset
X_train, y_train, X_val, y_val, X_test, y_test = get_datasets.get_datasets("merged_data.csv")

# Tensor conversion
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)  # Flatten y_train to match output dimensions
X_val = torch.tensor(X_test.values, dtype=torch.float32)
y_val = torch.tensor(y_test.values, dtype=torch.float32)  # Flatten y_test for the same reason
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)  # Flatten y_test for the same reason

# DataLoader setup
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=train_batch, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=test_batch, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=test_batch, shuffle=False)

# Model, optimizer and loss function
model = BinaryClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-04)
criterion = nn.BCELoss()

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

        predicted = torch.round(outputs).flatten()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100 * correct / total
    
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

            predicted = torch.round(outputs).flatten()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    val_loss = running_loss / len(val_loader.dataset)
    val_acc = 100 * correct / total


    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

torch.save(model.state_dict(), 'best_nn_adam.pth')

# Test set
y_pred = []
y_true = []
running_loss = 0.0

model.eval()
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predicted = torch.round(outputs).flatten()
        loss = criterion(outputs, targets.unsqueeze(1))
        running_loss += loss.item()

        y_pred.extend(predicted.numpy())
        y_true.extend(targets.numpy())

losses = running_loss / len(test_loader.dataset)

# Compute accuracy
acc = accuracy_score(y_true, y_pred)
print("Test Accuracy: ", acc)
# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)