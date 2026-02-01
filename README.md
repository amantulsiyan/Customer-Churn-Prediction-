# Customer-Churn-Prediction
# Customer Churn Prediction (Classification + Imbalanced Data)

## 1. Problem Statement

Customer churn refers to customers discontinuing a service. In subscription-based businesses, retaining customers is often cheaper than acquiring new ones. The objective of this project is to **predict whether a customer will churn** and to design the model in a way that prioritizes identifying churners.

This is framed as a **binary classification problem with class imbalance**, where the churn class (1) is the minority.

---

## 2. Dataset

* **Source**: Telco Customer Churn dataset
* **Target variable**: `Churn` (Yes / No → encoded as 1 / 0)
* **Key features**:

  * Demographics (gender, partner, dependents)
  * Account information (tenure, contract type, payment method)
  * Service usage (internet service, add-ons)
  * Billing information (monthly charges, total charges)

### Data Quality Issues Addressed

* `TotalCharges` stored as strings with blank values → converted to numeric and imputed with 0
* `customerID` removed as it is a unique identifier
* Categorical inconsistencies cleaned (whitespace, casing)

---

## 3. Preprocessing Pipeline

Preprocessing was implemented using **sklearn Pipelines and ColumnTransformer** to avoid data leakage and ensure reproducibility.

* Numerical features: `StandardScaler`
* Categorical features: `OneHotEncoder(handle_unknown='ignore', drop='first')`
* Target encoding: `No → 0`, `Yes → 1` (performed **before train-test split**)

---

## 4. Train–Test Split

* Split ratio: 80% train / 20% test
* Stratified split used to preserve churn distribution

---

## 5. Baseline Model

* **Model**: Logistic Regression
* **Reason**:

  * Strong baseline for binary classification
  * Interpretable coefficients
  * Works well with properly scaled features

### Baseline Results (Imbalanced Data)

* Accuracy ≈ 0.81
* Recall (Churn) ≈ 0.56

Although accuracy was high, the model missed many churners, making it unsuitable from a business perspective.

---

## 6. Handling Class Imbalance

Two approaches were evaluated:

### 6.1 Class Weighting (Chosen Approach)

* Used `class_weight='balanced'` in Logistic Regression
* Adjusts the loss function to penalize misclassification of churners more heavily

**Outcome**:

* Recall (Churn) improved significantly
* Accuracy dropped (expected trade-off)
* ROC-AUC remained stable

### 6.2 SMOTE (Evaluated, Not Chosen)

* Applied only on training data using `imblearn.Pipeline`
* Slight improvement in recall
* No meaningful reduction in false negatives compared to class weighting
* Added complexity and synthetic data

**Decision**: Rejected in favor of a simpler and more stable class-weighted model.

---

## 7. Threshold Tuning

Default classification threshold (0.5) was not optimal for churn detection.

Thresholds evaluated: **0.5, 0.4, 0.3**

| Threshold        | Recall (Churn) | False Negatives | Precision (Churn) |
| ---------------- | -------------- | --------------- | ----------------- |
| 0.5              | 0.78           | 81              | 0.51              |
| **0.4 (Chosen)** | **0.87**       | **50**          | 0.47              |
| 0.3              | 0.93           | 27              | 0.43              |

**Final threshold = 0.4**, chosen to balance high recall with acceptable business cost.

---

## 8. Evaluation Metrics

* Confusion Matrix
* Precision, Recall, F1-score
* ROC-AUC
* Precision–Recall Curve

### Final Model Performance

* Recall (Churn): **~0.87** (threshold tuned)
* ROC-AUC: **~0.84**

---

## 9. Business Interpretation

* False Negatives (missed churners) are costlier than False Positives
* Model is designed to catch as many churners as possible
* False positives represent customers offered unnecessary discounts, which is acceptable compared to losing customers

---

## 10. Final Model Summary

* Model: Logistic Regression
* Imbalance Handling: `class_weight='balanced'`
* Threshold: 0.4
* SMOTE: Evaluated and rejected

---

## 11. Key Learnings

* Accuracy is misleading for imbalanced datasets
* Recall and ROC-AUC are more informative for churn prediction
* Threshold tuning is as important as model selection
* Simpler models with clear business alignment are often preferable

---

## 12. Future Work

* Try tree-based models (Random Forest, XGBoost)
* Cost-sensitive optimization using explicit cost matrices
* Deployment with real-time inference and monitoring
