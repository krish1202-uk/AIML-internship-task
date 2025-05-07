# svm_classification.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("breast-cancer.csv")

# Convert diagnosis to numerical values (M = 1, B = 0)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Select features and target variable
X = df.iloc[:, 2:]  # Excluding id and diagnosis
y = df["diagnosis"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM with linear kernel
svm_linear = SVC(kernel="linear", C=1.0)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)

# Train SVM with RBF kernel
svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)

# Evaluate model performance
print("Linear Kernel Accuracy:", accuracy_score(y_test, y_pred_linear))
print("RBF Kernel Accuracy:", accuracy_score(y_test, y_pred_rbf))
print("\nClassification Report (RBF Kernel):\n", classification_report(y_test, y_pred_rbf))
print("\nConfusion Matrix (RBF Kernel):\n", confusion_matrix(y_test, y_pred_rbf))

# Visualize decision boundary using PCA-reduced 2D data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train SVM with PCA-reduced data
svm_pca = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_pca.fit(X_pca, y_train)

# Plot decision boundary
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap="coolwarm", edgecolors="k", alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("SVM Decision Boundary (PCA Reduced)")
plt.show()

# Hyperparameter tuning using cross-validation
C_values = [0.1, 1, 10, 100]
gamma_values = ["scale", 0.1, 1, 10]

best_accuracy = 0
best_params = {}

for C in C_values:
    for gamma in gamma_values:
        model = SVC(kernel="rbf", C=C, gamma=gamma)
        scores = cross_val_score(model, X_train, y_train, cv=5)
        mean_accuracy = np.mean(scores)

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = {"C": C, "gamma": gamma}

print("\nBest Hyperparameters:", best_params)
print("Best Cross-validation Accuracy:", best_accuracy)
