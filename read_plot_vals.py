import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data from a CSV file
df = pd.read_csv('./svm_counterfactual.csv')
df = df[df['Change Frequency (%)'] >= 0.2] #filtering df for significant changes
# Normalize the frequency data for coloring
df['Normalized Frequency'] = (df['Change Frequency (%)'] - df['Change Frequency (%)'].min()) / (df['Change Frequency (%)'].max() - df['Change Frequency (%)'].min())

# Create a color map based on 'Change Frequency (%)'
norm = plt.Normalize(df['Normalized Frequency'].min(), df['Normalized Frequency'].max())
cmap = sns.color_palette("coolwarm", as_cmap=True)
colors = cmap(norm(df['Normalized Frequency']))

# Create the plot
plt.figure(figsize=(10, 8))
sns.swarmplot(x='Average Change (%)', y='Feature', data=df, size=10, palette=colors)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, label='Change Frequency (%)')

# Set title and labels
plt.title('Bee-Swarm Plot of Feature Changes in Counterfactual Analysis')
plt.xlabel('Average Change (%)')
plt.ylabel('Feature')

plt.show()