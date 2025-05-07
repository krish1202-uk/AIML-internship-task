import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Step 1: Load the Dataset

df = pd.read_csv("Titanic-Dataset.csv")

# Display basic info
print("First five rows:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())


# Step 2: Visualize Data with Histograms & Boxplots for Numeric Features

# Identify numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Histograms for each numeric feature
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Boxplots for each numeric feature
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()


# Step 3: Explore Feature Relationships with Pairplots and Correlation Matrix

# Pairplot for numeric features to spot any relationships or clusters
sns.pairplot(df[numeric_cols])
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.show()

# Compute correlation matrix and display as a heatmap
corr_matrix = df[numeric_cols].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()


# Step 4: Interactive Visualization using Plotly to Identify Patterns or Anomalies

# Create an interactive scatter plot using Plotly to examine the relationship 
# between Age and Fare, colored by 'Survived'.
fig = px.scatter(
    df, 
    x="Age", 
    y="Fare", 
    color="Survived", 
    title="Interactive Scatter Plot: Age vs Fare",
    labels={"Age": "Passenger Age", "Fare": "Ticket Fare"},
    hover_data=["Name"]
)
fig.show()



