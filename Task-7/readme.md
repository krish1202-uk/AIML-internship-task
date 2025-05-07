

📍 **Objective:**  
Use SVMs for linear and non-linear classification on a dataset.

📂 **Dataset:**  
- File: `breast-cancer.csv`  
- Target variable: `diagnosis` (🦠 Malignant vs ✅ Benign)

🛠️ **Tools:**  
✅ Scikit-learn  
✅ NumPy  
✅ Matplotlib  
✅ Pandas  

🔹 **Steps:**  
1️⃣ **Load and Preprocess Data**  
   🔹 Convert target variable to numerical values.  
   🔹 Standardize features using `StandardScaler`.  

2️⃣ **Train SVM Models**  
   🔹 Fit models with **Linear** and **RBF** kernels.  
   🔹 Evaluate performance using accuracy.  

3️⃣ **Visualize Decision Boundary**  
   🔹 Apply **PCA** for 2D feature representation.  
   🔹 Plot data points with decision boundary.  

4️⃣ **Hyperparameter Tuning**  
   🔹 Use **cross-validation** to find best `C` and `gamma` values.  

📊 **Evaluation:**  
✅ Classification accuracy of models  
✅ Confusion matrix & classification report  
✅ PCA visualization  
✅ Best hyperparameters using cross-validation  

🚀 **How to Run:**  
💾 Save the script as `prog.py`.  
▶️ Run:
   ```sh
   python prog.py
