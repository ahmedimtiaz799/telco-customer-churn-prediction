# Telco Customer Churn Prediction 📞📉

This project focuses on predicting telecom customer churn using machine learning. The Telco Customer Churn dataset from Kaggle is used to train models that identify customers likely to leave the service. Techniques such as feature engineering, class balancing, and threshold tuning were applied to improve prediction performance.

## 📂 Dataset

- **Source:** Kaggle – Telco Customer Churn  
- **Records:** 7,043  
- **Target Variable:** Churn (Yes = customer left, No = customer stayed)  
- **Feature Types:** Demographics, account info, subscribed services, contract type, and billing details  

## ⚙️ Technologies Used

- Python  
- Pandas, NumPy  
- Seaborn, Matplotlib  
- Scikit-learn  
- Jupyter Notebook  

## 🔍 Project Workflow

### 1. Data Loading & Initial Exploration
- Loaded dataset using `pandas.read_csv()`
- Explored data using `.head()`, `.info()`, `.shape`, and `.columns`

### 2. Data Cleaning
- Removed duplicate records  
- Handled missing values in the `TotalCharges` column  
- Analyzed class distribution in the target variable `Churn`  

### 3. Exploratory Data Analysis (EDA)
- Plotted churn distribution with `countplot()`  
- Histograms for `tenure`, `MonthlyCharges`, and `TotalCharges`  
- Boxplots to compare churn behavior with numeric features  
- Correlation barplot to identify strong features related to churn  

### 4. Feature Engineering
- One-Hot Encoded categorical service features  
- Label Encoded binary variables  
- Applied StandardScaler to `tenure`, `MonthlyCharges`, and `TotalCharges`  
- Final features included service combinations, contract types, and payment methods  

### 5. Model Training
- **Features (X):** All columns except `Churn`  
- **Target (y):** `Churn`  
- Train-test split performed with stratification  

#### Logistic Regression
- Trained with `class_weight='balanced'`  
- Initial results showed low recall for churners  
- Threshold tuning (e.g., 0.45) improved churn detection slightly  

#### Random Forest Classifier (Tuned)
- Hyperparameter tuning using `GridSearchCV`  
- Tuned: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weight`  
- Final model provided better performance on minority class  

## 📈 Evaluation Metrics

- Classification Report (Precision, Recall, F1 Score)  
- Confusion Matrix  
- Accuracy  
- Threshold Tuning for better churn class recall  

| Model                  | F1-Score (Churn) | Precision (Churn) | Recall (Churn) |
|------------------------|------------------|--------------------|----------------|
| Logistic Regression    | 0.41             | 0.61               | 0.30           |
| Random Forest (tuned)  | 0.61             | 0.52               | 0.75           |

## 📊 Visualizations

- Churn distribution (countplot)  
- Histograms of `MonthlyCharges`, `TotalCharges`, `tenure`  
- Boxplots grouped by churn status  
- Feature correlation barplot  

## ✅ Results Summary

- **Best Model:** Tuned Random Forest  
- **Threshold Used:** 0.45  
- **Accuracy:** 75%  
- **Churn Recall:** 75%  
- **Precision (Churn):** 52%  
- The model prioritized recall to effectively capture potential churners for retention action  
