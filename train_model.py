"""
AI Risk Manager - Fraud-Spike Detector
Razorpay Buildathon - Track 02

Goal: Detect fraudulent transactions with measured precision/recall,
and quantify the real business cost of false positives vs false negatives.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
    precision_recall_curve
)
import joblib

print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)
df = pd.read_csv('data/creditcard.csv')
print(f"Total transactions: {len(df):,}")
print(f"Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")

# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Feature prep")
print("=" * 60)
# Scale 'Amount' and 'Time' - the V1-V28 are already PCA-scaled
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])

feature_cols = [c for c in df.columns if c not in ['Time', 'Amount', 'Class']]
X = df[feature_cols]
y = df['Class']

# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Train/test split (stratified - preserves fraud ratio)")
print("=" * 60)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Fraud in test set: {y_test.sum()}")

# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Train model (Random Forest, class_weight=balanced)")
print("=" * 60)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight='balanced',   # <-- key: tells model to care more about rare fraud class
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Model trained.")

# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Evaluate at default threshold (0.5)")
print("=" * 60)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")
print(f"\nConfusion Matrix:")
print(f"  True Negatives (correctly cleared):  {tn:,}")
print(f"  False Positives (flagged genuine):   {fp:,}  <- annoys good customers")
print(f"  False Negatives (missed fraud):      {fn:,}  <- costs Razorpay/merchant money")
print(f"  True Positives (caught fraud):       {tp:,}")

# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Business cost analysis (THE DIFFERENTIATOR)")
print("=" * 60)
# Assumptions - clearly state these are illustrative for the demo
COST_FALSE_POSITIVE = 5      # $ - support cost + customer friction/churn risk per wrongly-flagged txn
avg_fraud_amount = df[df['Class'] == 1]['Amount'].mean()
print(f"Average fraud transaction amount: ${avg_fraud_amount:.2f}")
print(f"Assumed cost per false positive (customer friction/support): ${COST_FALSE_POSITIVE}")

cost_fp = fp * COST_FALSE_POSITIVE
cost_fn = fn * avg_fraud_amount   # money lost outright to fraud

print(f"\nTotal cost from False Positives (wrongly blocked genuine users): ${cost_fp:,.2f}")
print(f"Total cost from False Negatives (fraud that slipped through):    ${cost_fn:,.2f}")
print(f"Combined cost at default threshold (0.5):                       ${cost_fp + cost_fn:,.2f}")

# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: Find optimal threshold (minimize business cost, not just F1)")
print("=" * 60)
thresholds = np.arange(0.05, 0.95, 0.05)
results = []
for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    cm_t = confusion_matrix(y_test, y_pred_t)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    total_cost = (fp_t * COST_FALSE_POSITIVE) + (fn_t * avg_fraud_amount)
    results.append({
        'threshold': t, 'fp': fp_t, 'fn': fn_t, 'tp': tp_t,
        'precision': precision_score(y_test, y_pred_t, zero_division=0),
        'recall': recall_score(y_test, y_pred_t, zero_division=0),
        'total_cost': total_cost
    })

results_df = pd.DataFrame(results)
best_row = results_df.loc[results_df['total_cost'].idxmin()]
print(results_df.round(2).to_string(index=False))
print(f"\n>>> Optimal threshold to minimize business cost: {best_row['threshold']:.2f}")
print(f">>> At this threshold -> Precision: {best_row['precision']:.3f}, "
      f"Recall: {best_row['recall']:.3f}, Cost: ${best_row['total_cost']:,.2f}")
print(f">>> Savings vs default threshold: ${(cost_fp+cost_fn) - best_row['total_cost']:,.2f}")

# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 8: Save model + results")
print("=" * 60)
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
results_df.to_csv('threshold_analysis.csv', index=False)
print("Saved: fraud_model.pkl, scaler.pkl, threshold_analysis.csv")
print("\nDONE.")
