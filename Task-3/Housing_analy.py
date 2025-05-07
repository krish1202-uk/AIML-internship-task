import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load & Preprocess Data  
def load_data():
    """Loads the housing dataset and preprocesses it."""
    df = pd.read_csv("housing.csv")  # Load dataset
    
    # Drop missing values (if any)
    df.dropna(inplace=True)

    # Selecting relevant features based on analysis
    X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]  # Predictors
    y = df['price']  # Target variable (house price)
    
    return X, y

# Step 2️: Split Data into Train-Test Sets  
def split_data(X, y):
    """Splits the data into training and testing sets."""
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3️: Train Linear Regression Model  
def train_model(X_train, y_train):
    """Trains a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Step 4️: Evaluate Model Performance  
def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model using common metrics."""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f" Model Performance:\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nR² Score: {r2:.4f}")
    return y_pred

# Step 5️: Visualize Regression Results  
def plot_regression(X_test, y_test, y_pred):
    """Plots the regression line to visualize results."""
    plt.scatter(X_test['area'], y_test, color='blue', label="Actual Prices")  # Using area as key feature
    plt.plot(X_test['area'], y_pred, color='red', linewidth=2, label="Predicted Prices")
    
    plt.xlabel("Area (sq ft)")
    plt.ylabel("Price ($)")
    plt.title("Housing Price Prediction - Linear Regression")
    plt.legend()
    plt.show()

#  Execute the Complete Pipeline  
if __name__ == "__main__":
    print(" Loading & Preprocessing Data...")
    X, y = load_data()

    print("✂ Splitting Data into Train-Test Sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(" Training Linear Regression Model...")
    model = train_model(X_train, y_train)

    print(" Evaluating Model Performance...")
    y_pred = evaluate_model(model, X_test, y_test)

    print(" Visualizing Regression Results...")
    plot_regression(X_test, y_test, y_pred)

    print(" Analysis Complete! ")
