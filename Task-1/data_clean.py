import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load CSV Dataset
df = pd.read_csv("data.csv")  # Replace with actual CSV filename

# Show basic info
print("Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nFirst few rows:")
print(df.head())

# Step 2: Handle Missing Values
numeric_cols = df.select_dtypes(include=['number']).columns  # Numeric columns
categorical_cols = df.select_dtypes(include=['object']).columns  # Categorical columns

# Fill numeric missing values with mean
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()))

# Fill categorical missing values with mode
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Step 3: Encoding Categorical Features
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes  # Convert categorical to numeric

# Step 4: Visualizing Outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numeric_cols])
plt.title("Outliers Visualization")
plt.show()

# Step 5: Removing Outliers using IQR
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Step 6: Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)
print("\nCleaned data saved as 'cleaned_dataset.csv'!")
