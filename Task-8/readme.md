💠 **Clustering with K-Means**  

📍 **Objective:**  
Perform unsupervised clustering on customer data using K-Means.

📂 **Dataset:**  
- File: `Mall_Customers.csv`  
- Features: **Annual Income** & **Spending Score**  

🛠️ **Tools:**  
✅ Scikit-learn  
✅ Pandas  
✅ Matplotlib  
✅ NumPy  

🔹 **Steps:**  
1️⃣ **Load and Preprocess Data**  
   🔹 Select numerical features for clustering.  
   🔹 Standardize features using `StandardScaler`.  

2️⃣ **Elbow Method for Optimal K**  
   🔹 Plot inertia values for clusters **K = 1 to 10**.  

3️⃣ **Fit K-Means Model**  
   🔹 Train model with the chosen number of clusters (`K`).  
   🔹 Assign cluster labels to customers.  

4️⃣ **Evaluate Clustering**  
   🔹 Compute **Silhouette Score** for clustering quality.  

5️⃣ **Visualize Clusters**  
   🔹 Scatter plot of clusters with color coding.  

🚀 **How to Run:**  
💾 Save the script as `prog.py`.  
▶️ Run:
   ```sh
   python prog.py
S