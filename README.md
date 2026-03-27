# 💳 Credit Card Fraud Detection

## 📌 Project Overview
This project detects fraudulent credit card transactions using Machine Learning. It handles imbalanced data using SMOTE and compares models like Random Forest and XGBoost.

---

## 📊 Dataset
- Source: Kaggle Credit Card Fraud Detection Dataset
- Total Records: 284,807
- Features: 30
- Target: Class (0 = Normal, 1 = Fraud)

---

## ⚙️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost

---

## 🔍 Steps

1. Data Loading & EDA  
2. Data Visualization  
3. SMOTE for imbalance handling  
4. Model Training (RF, XGBoost)  
5. Evaluation (Precision, Recall, ROC-AUC)  

---

## 📈 Results
- Successfully detects fraud transactions  
- XGBoost performs better on imbalanced data  

---

## 🚀 How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
python fraud_project.py
