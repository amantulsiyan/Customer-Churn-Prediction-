#Imports
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline 

#Load Dataset & EDA
df=pd.read_csv("Customer Churn Dataset.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"]=df["TotalCharges"].fillna(0)
df["Contract"]=df["Contract"].str.strip()
df["Contract"]=df["Contract"].str.lower()
df["Churn"]=df["Churn"].map({"No":0,"Yes":1})
X=df.drop(columns=['customerID','Churn'],axis=1)
y=df["Churn"]

# Train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

#Preprocessing
categorical_cols=X.select_dtypes(include="object").columns
numerical_cols=X.select_dtypes(exclude="object").columns

preprocessor=ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat",OneHotEncoder(handle_unknown='ignore',drop="first"),categorical_cols)
    ]
)

#Model Training
model=Pipeline(
    steps=[
        ("preprocess",preprocessor),
        ("classifier", LogisticRegression(max_iter=1000,class_weight="balanced"))
    ]
)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
y_prob=model.predict_proba(X_test)[:,1]

#Evaluation 
accuracy=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
report=classification_report(y_test,y_pred)
roc_auc=roc_auc_score(y_test,y_prob)

print("\nAccuracy Score(LR+ Class weight='balanced')\n",accuracy)
print("\nConfusion Matrix(LR+ Class weight='balanced')\n",cm)
print("Classification report(LR+ Class weight='balanced'):\n",report)
print("ROC(Receiver Operating Characteristic)(LR+ Class weight='balanced'):\n",roc_auc)

#ROC Curve
fpr, tpr, _=roc_curve(y_test,y_prob)
plt.figure(figsize=(10,6))
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

#Threshold Analysis
for t in [0.5,0.4,0.3]:
    y_pred_t=(y_prob>=t).astype(int)
    print(f"For threshold t ={t}")
    print("Threshold Tuned metrics:")
    print("Classification Report:\n",classification_report(y_test,y_pred_t))
    print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred_t),"\n")

#Recall vs Precision Curve
precision,recall,thresholds=precision_recall_curve(y_test,y_prob)
plt.figure(figsize=(10,6))
plt.plot(recall,precision)
plt.xlabel("Recall Score")
plt.ylabel("Precision Score")
plt.title("Recall vs Precision Score")
plt.grid(True)
plt.show()

smote=SMOTE(random_state=42)

model_smote=ImbPipeline(
    steps=[
        ("preprocess",preprocessor),
        ("smote",SMOTE()),
        ("classifier",LogisticRegression(max_iter=1000))
    ]
)

model_smote.fit(X_train,y_train)
y_pred_smote=model_smote.predict(X_test)
y_prob_smote=model_smote.predict_proba(X_test)[:,1]

accuracy_smote=accuracy_score(y_test,y_pred_smote)
cm_smote=confusion_matrix(y_test,y_pred_smote)
report_smote=classification_report(y_test,y_pred_smote)
roc_auc_smote=roc_auc_score(y_test,y_prob_smote)

print("\nAccuracy Score(SMOTE)\n",accuracy_smote)
print("\nConfusion Matrix(SMOTE)\n",cm_smote)
print("Classification report(SMOTE):\n",report_smote)
print("ROC(Receiver Operating Characteristic)(SMOTE):\n",roc_auc_smote)