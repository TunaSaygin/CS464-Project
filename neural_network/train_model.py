import torch
from model import Model
import matplotlib.pyplot as plt
import os
import pandas as pd

# get the values
current_path = os.getcwd()
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))

file_path_x = f'{parent_path}/data_imputed.csv'
file_path_y = f'{parent_path}/data01.csv'

data_x = pd.read_csv(file_path_x)
data_y = pd.read_csv(file_path_y)

# Drop the 'ID' column from the dataframe
data_x = data_x.drop(columns=['ID'])
data_y = data_y[["outcome"]]

# Convert the DataFrame to a tensor
X = torch.tensor(data_x.values, dtype=torch.float32)
y = torch.tensor(data_y.values, dtype=torch.float32)

# define variables
num_of_epochs = 1000
n_features = 49
n_nodes = 98

# define the model
model = Model(n_features, n_nodes)

optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()

# fit the model
for i in range(num_of_epochs):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y, y_pred)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(loss)


# plot the graphs
x = torch.linspace(-5, 5, 100).reshape(-1, 1)
x = torch.hstack(n_features*[x])

for i in range(n_features):
    plt.plot(
        x[:, 0].detach().numpy(),
        model.get_submodule('lr').weight[0][i].item() * model.get_submodule('features')(x)[:, i].detach().numpy())
    plt.title(f'Feature {i+1}')
    plt.show()
