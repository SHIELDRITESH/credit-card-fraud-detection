import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("creditcard.csv")

print(df.head())

print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df['Class'].value_counts())
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Normal Transactions")
plt.show()
from sklearn.model_selection import train_test_split

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(y_train_smote.value_counts())

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train_smote, y_train_smote)

from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train_smote, y_train_smote)

y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)
from sklearn.metrics import classification_report

print("Random Forest:\n", classification_report(y_test, y_pred_rf))
print("XGBoost:\n", classification_report(y_test, y_pred_xgb))

from sklearn.metrics import roc_auc_score

rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1])

print("RF ROC-AUC:", rf_auc)
print("XGB ROC-AUC:", xgb_auc)

y_probs = rf.predict_proba(X_test)[:,1]

y_pred_custom = (y_probs > 0.3).astype(int)

print(classification_report(y_test, y_pred_custom))