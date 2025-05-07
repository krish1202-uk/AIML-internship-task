# 🌳 Decision Trees & 🌲 Random Forests for Heart Disease Prediction

## 📌 Objective
In this project, we use **tree-based machine learning models** to predict heart disease risk, comparing the effectiveness of **Decision Trees** and **Random Forests**. 🏥💙  

## 🔧 Tools & Libraries  
✔ **Scikit-learn** – Machine Learning Toolkit 🧠  
✔ **Matplotlib** – Data Visualization 📊  
✔ **Graphviz** – Tree Structure Rendering 🌐  
✔ **Pandas & NumPy** – Data Handling & Processing 📑  

## 🚀 Workflow & Steps  
### 1️⃣ **Data Preparation**  
📂 Load `heart.csv` dataset.  
🔍 Split data into training (80%) and testing (20%) sets.  
⚙ Define feature columns & target labels.  


2️⃣ **Decision Tree Model**  
🌳 Train `DecisionTreeClassifier` (max_depth=4) 🏗  
📊 Visualize tree structure to analyze decision logic.  
✅ Evaluate accuracy using test data.  

### 3️⃣ **Overfitting Analysis & Depth Control**  
🔍 Limit tree depth for **better generalization** 📈  
⚖ Examine accuracy changes across depths.  

### 4️⃣ **Random Forest Model**  
🔥 Train `RandomForestClassifier` (100 trees).  
⚡ Compare accuracy with Decision Tree.  
🎯 Perform **cross-validation** for robustness.  

### 5️⃣ **Feature Importance Analysis**  
📊 Identify most influential factors in predictions.  
🎨 Visualize feature importance rankings.  

## 📏 Evaluation Metrics  
🔹 **Accuracy Score** – Measures correctness ✅  
🔹 **Cross-validation** – Ensures reliability 🔄  

## 🎯 Key Findings  
✔ **Random Forests outperform Decision Trees** due to ensemble learning! 🏆  
✔ Limiting Decision Tree depth **reduces overfitting** ✅  
✔ **Feature Importance Analysis** helps understand critical predictors.  

## 🎭 Conclusion  
🌟 Decision Trees offer **interpretability**, but Random Forests ensure **better accuracy** through ensemble power.  
🔮 Future improvements: **hyperparameter tuning**, **ensemble stacking**, and **deep learning integration**! 🚀  


💡 **Pro Tip:** Always test different hyperparameters to optimize tree-based models for the best results! 🔥  
