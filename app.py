"""
AI Risk Manager - Fraud-Spike Detector Demo
Razorpay Buildathon Submission
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score

st.set_page_config(page_title="AI Risk Manager - Fraud Detector", layout="wide")

st.title("🛡️ AI Risk Manager — Fraud-Spike Detector")
st.caption("Razorpay Buildathon · Track 02 · Detecting fraud with business-cost-optimized thresholds")

# ---------- Load everything ----------
@st.cache_resource
def load_data_and_model():
    df = pd.read_csv('data/creditcard.csv')
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])
    feature_cols = [c for c in df.columns if c not in ['Time', 'Amount', 'Class']]
    X = df[feature_cols]
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    model = joblib.load('fraud_model.pkl')
    y_proba = model.predict_proba(X_test)[:, 1]
    return df, X_test, y_test, y_proba

df, X_test, y_test, y_proba = load_data_and_model()
avg_fraud_amount = df[df['Class'] == 1]['Amount'].mean()

# ---------- Sidebar controls ----------
st.sidebar.header("⚙️ Decision Settings")
threshold = st.sidebar.slider(
    "Fraud probability threshold", 0.05, 0.95, 0.15, 0.05,
    help="Transactions with fraud-probability above this get flagged."
)
cost_fp = st.sidebar.number_input(
    "Cost per false positive ($)", value=5.0, step=1.0,
    help="Support cost + customer friction from wrongly blocking a genuine transaction."
)

# ---------- Compute at chosen threshold ----------
y_pred = (y_proba >= threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
total_cost = (fp * cost_fp) + (fn * avg_fraud_amount)

# ---------- Metrics row ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precision", f"{precision:.1%}")
col2.metric("Recall", f"{recall:.1%}")
col3.metric("Fraud Caught", f"{tp} / {tp+fn}")
col4.metric("Estimated Business Cost", f"${total_cost:,.0f}")

st.divider()

# ---------- Confusion matrix breakdown ----------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual: Genuine", "Actual: Fraud"],
        columns=["Predicted: Genuine", "Predicted: Fraud"]
    )
    st.dataframe(cm_df, use_container_width=True)
    st.caption(
        f"🟠 {fp} genuine customers wrongly flagged (friction cost) · "
        f"🔴 {fn} fraud cases missed (money lost)"
    )

with c2:
    st.subheader("Cost Breakdown")
    st.write(f"**False Positive cost:** {fp} × ${cost_fp:.0f} = ${fp*cost_fp:,.0f}")
    st.write(f"**False Negative cost:** {fn} × ${avg_fraud_amount:.0f} (avg fraud amt) = ${fn*avg_fraud_amount:,.0f}")
    st.write(f"**Total cost at threshold {threshold}:** ${total_cost:,.0f}")

st.divider()

# ---------- Cost curve ----------
st.subheader("📉 Business Cost vs Threshold (find the sweet spot)")
thresholds = np.arange(0.05, 0.95, 0.05)
rows = []
for t in thresholds:
    yp = (y_proba >= t).astype(int)
    cmt = confusion_matrix(y_test, yp)
    tnt, fpt, fnt, tpt = cmt.ravel()
    rows.append({
        "threshold": t,
        "total_cost": (fpt * cost_fp) + (fnt * avg_fraud_amount)
    })
curve_df = pd.DataFrame(rows).set_index("threshold")
st.line_chart(curve_df)

best_t = curve_df['total_cost'].idxmin()
st.success(f"💡 Optimal threshold to minimize cost: **{best_t:.2f}** "
           f"(cost: ${curve_df['total_cost'].min():,.0f})")

st.divider()
st.caption(
    "Model: Random Forest (class_weight=balanced) on the Kaggle Credit Card Fraud dataset "
    "(284,807 txns, 0.17% fraud rate). Business-cost framing is illustrative for demo purposes."
)
