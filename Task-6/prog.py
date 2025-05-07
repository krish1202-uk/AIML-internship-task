import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Load the dataset
df = pd.read_csv("Iris.csv")

# Drop the ID column as it’s not useful for classification
df.drop(columns=["Id"], inplace=True)

# Convert categorical labels to numeric
df["Species"] = df["Species"].astype("category").cat.codes

# Define features and target
X = df.drop(columns=["Species"])
y = df["Species"]

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize feature values for better KNN performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN classifier with different values of K
k_values = [3, 5, 7]  # Testing different K values
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    # Predictions & accuracy
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"KNN Accuracy for k={k}: {acc:.2f}")

# Best K selection (optional)
best_k = 5  # Choose optimal K based on accuracy results
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
y_pred_best = knn_best.predict(X_test_scaled)

# Confusion Matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=df["Species"].astype("category").cat.categories,
            yticklabels=df["Species"].astype("category").cat.categories)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix (K={best_k})")
plt.show()

print("Task completed successfully!")
