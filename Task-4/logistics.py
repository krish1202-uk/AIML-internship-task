import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

# Load and preprocess the dataset
def load_data():
    df = pd.read_csv("data.csv")  # Load the dataset

    # Ensure the dataset is not empty
    if df.empty:
        raise ValueError("The dataset is empty or not loading correctly.")

    # Check if the required columns exist
    required_columns = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' is missing in the dataset.")

    # Convert categorical target (M/B) to binary (0 = Benign, 1 = Malignant)
    df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
    
    # Select relevant features and target
    X = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']]  
    y = df['diagnosis']  
    return X, y

# Split the data into training and testing sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
def standardize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Train the Logistic Regression model
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Model Performance:")
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"ROC-AUC Score: {roc_auc:.2f}")
    print("Confusion Matrix:")
    print(cm)

# Main execution
if __name__ == "__main__":
    try:
        X, y = load_data()
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
        model = train_model(X_train_scaled, y_train)
        evaluate_model(model, X_test_scaled, y_test)
    except ValueError as e:
        print(f"Error: {e}")
