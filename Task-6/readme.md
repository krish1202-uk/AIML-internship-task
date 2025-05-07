# 🌿 K-Nearest Neighbors (KNN) Classification on Iris Dataset 🌟

## 📌 Objective
This project implements the **K-Nearest Neighbors (KNN) algorithm** to classify species in the **Iris dataset**, exploring the impact of different values of **K** and evaluating model performance. 🌸  

## 🛠️ Tools & Libraries  
🔹 **Scikit-learn** – Machine Learning Toolkit 🤖  
🔹 **Pandas & NumPy** – Data Handling & Processing 📊  
🔹 **Matplotlib & Seaborn** – Visualization Tools 🎨  

## 🚀 Workflow & Steps  
### 1️⃣ **Data Preparation**  
📥 Load `Iris.csv` dataset.  
📊 Remove unnecessary columns (`Id`).  
🔄 Convert categorical labels (`Species`) into numerical format.  

### 2️⃣ **Feature Scaling**  
🎯 Normalize feature values for better KNN accuracy using **StandardScaler**.  

### 3️⃣ **Train KNN Classifier**  
⚙ Train **KNeighborsClassifier** with different values of `K` (`3, 5, 7`).  
📌 Evaluate accuracy for each `K` and select the optimal value.  

### 4️⃣ **Model Evaluation**  
✔ Measure classification **accuracy**.  
🔍 **Confusion Matrix** visualization using **Seaborn heatmap**.  

## 📏 Evaluation Metrics  
🔹 **Accuracy Score** – Measures correctness ✅  
🔹 **Confusion Matrix** – Analyzes classification performance 📈  

## 🎯 Key Findings  
✔ **Feature scaling improves KNN performance** by normalizing input values.  
✔ **Optimal K selection** ensures better classification accuracy 🎯.  
✔ **Confusion Matrix** helps analyze misclassifications effectively.  
