import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



#Load Dataset
df=pd.read_csv("2.Customer Churn Prediction (Classification + Imbalanced Data)\WA_Fn-UseC_-Telco-Customer-Churn.csv")
"""
print(df.isnull().sum())
print(df.dtypes)
"""
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"]=df["TotalCharges"].fillna(0)
df["Contract"]=df["Contract"].str.strip()
df["Contract"]=df["Contract"].str.lower()
y=df["Churn"].map({"No":0,"Yes":1})
X=df.drop(columns=['customerID','Churn'],axis=1)
y=df["Churn"]

# Train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(y_test.dtypes)
#One Hot Encoding
categorical_cols=X.select_dtypes(include="object").columns
numerical_cols=X.select_dtypes(exclude="object").columns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
preprocessor=ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat",OneHotEncoder(handle_unknown='ignore',drop="first"),categorical_cols)
    ]
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

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

#accuracy evaluation 
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("\nAccuracy Score\n",accuracy)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix\n",cm)


###We didnt use the following code because of encoding problems and instaed used the one below it for better clarity and representation purpose
"""from sklearn.metrics import precision_score, recall_score, f1_score
precision=precision_score(y_test, y_pred)
recall=recall_score(y_test, y_pred)
f1=f1_score(y_test, y_pred) 
print("\nPrecision Score\n",precision)
print("\nRecall Score\n",recall)
print("\nF1 Score:\n",f1)
"""
from sklearn.metrics import classification_report
print("Classification report:\n",classification_report(y_test,y_pred))

from sklearn.metrics import roc_auc_score
roc=roc_auc_score(y_test,y_prob)
print("ROC(Receiver Operating Characteristic):\n",roc)

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt 
fpr, tpr, _=roc_curve(y_test,y_prob)
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("Total Positive Rate")
plt.title("ROC Curve")
plt.plot()



