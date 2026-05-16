import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data exported from R
df = pd.read_csv(
    r"D:\Articles@arf\ML\ForecastingLab\final_data2.csv"
)
print(df.head())

# Remove time column
X = df.drop(columns=["time"])

# Convert everything to numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

# Keep variables with <35% missing
X = X.loc[:, X.isna().mean() < 0.35]

# Linear interpolation
X = X.interpolate(method="linear")

# Remove remaining NA
X = X.dropna()

# Standardize
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()

scores = pca.fit_transform(X_scaled)

# Explained variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Scree plot
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_,
    marker="o"
)

plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")

plt.grid(True)

plt.show()

# Loadings
loadings = pd.DataFrame(
    pca.components_.T,
    index=X.columns,
    columns=[
        f"PC{i+1}"
        for i in range(len(X.columns))
    ]
)

print("\nPCA Loadings:")
print(loadings.iloc[:, 0:3].round(2))

# PCA factors
pc_df = pd.DataFrame({
    "PC1": scores[:, 0],
    "PC2": scores[:, 1]
})

print("\nPCA Factors:")
print(pc_df.head())

# Plot PC1
plt.figure(figsize=(10, 5))

plt.plot(pc_df["PC1"])

plt.title("PC1")
plt.xlabel("Observation")
plt.ylabel("PC1")

plt.grid(True)

plt.show()
