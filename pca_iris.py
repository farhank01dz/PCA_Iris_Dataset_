import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable (species)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for visualization
df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
df['Target'] = y

# Plotting the PCA result
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
target_names = iris.target_names

for i, color in zip(range(len(target_names)), colors):
    plt.scatter(df[df['Target'] == i]['Principal Component 1'],
                df[df['Target'] == i]['Principal Component 2'],
                color=color, label=target_names[i], edgecolor='k', s=100)

plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.show()
