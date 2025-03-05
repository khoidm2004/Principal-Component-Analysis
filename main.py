import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

# Load Wine Dataset
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Standardize the Data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components
principal_components = pca.fit_transform(df_scaled)

# Create DataFrame for visualization
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Target'] = data.target

# Explained variance
explained_variance = pca.explained_variance_ratio_
print(f'Explained Variance Ratio: {explained_variance}')

# Plot PCA results
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue=pca_df['Target'], palette='Set1', data=pca_df)
plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)')
plt.title('PCA of Wine Dataset')
plt.legend(title='Wine Class')
plt.show()

# Bonus: PCA for Image Compression
from sklearn.datasets import load_digits

digits = load_digits()
X_digits = digits.data

# Standardize the data
X_digits_scaled = StandardScaler().fit_transform(X_digits)

# Apply PCA
pca_digits = PCA(n_components=16)  # Reduce dimensionality to 16 components
X_pca = pca_digits.fit_transform(X_digits_scaled)

# Inverse transform to reconstruct the image
X_reconstructed = pca_digits.inverse_transform(X_pca)

# Plot original vs reconstructed images
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(digits.images[0], cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(X_reconstructed[0].reshape(8, 8), cmap='gray')
axes[1].set_title('Reconstructed with PCA')
plt.show()
