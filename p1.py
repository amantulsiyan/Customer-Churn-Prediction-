
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
    roc_auc_score
)

#Load Dataset & EDA

df=pd.read_csv("2.Customer Churn Prediction (Classification + Imbalanced Data)\WA_Fn-UseC_-Telco-Customer-Churn.csv")

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
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
y_prob=model.predict_proba(X_test)[:,1]
print("Predicted Values\n",y_pred)

#Evaluation 

accuracy=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
classification_report=classification_report(y_test,y_pred)
roc=roc_auc_score(y_test,y_prob)

print("\nAccuracy Score\n",accuracy)
print("\nConfusion Matrix\n",cm)
print("Classification report:\n",classification_report)
print("ROC(Receiver Operating Characteristic):\n",roc)

fpr, tpr, _=roc_curve(y_test,y_prob)
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()




