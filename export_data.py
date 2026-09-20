"""Run once: python export_data.py  ->  writes frontend/public/data.json
Same split/scaling logic as app.py, so numbers match your Streamlit app."""
import json
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

df = pd.read_csv('data/creditcard.csv')
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])
cols = [c for c in df.columns if c not in ['Time', 'Amount', 'Class']]
X, y = df[cols], df['Class']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
model = joblib.load('fraud_model.pkl')
p = model.predict_proba(X_test)[:, 1]

rows = []
for i in range(1, 100):  # thresholds 0.01 .. 0.99
    t = i / 100
    tn, fp, fn, tp = confusion_matrix(y_test, (p >= t).astype(int), labels=[0, 1]).ravel()
    rows.append(dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))

amt = df.loc[X_test.index, 'Amount'].values
top = np.argsort(-p)[:14]
samples = [dict(id=f"TXN-{int(X_test.index[j]):06d}", amount=round(float(amt[j]), 2),
                p=round(float(p[j]), 3), y=int(y_test.iloc[j])) for j in top]

out = dict(avg_fraud=float(df[df.Class == 1].Amount.mean()), n_test=len(y_test),
           n_fraud=int(y_test.sum()), rows=rows, samples=samples)
json.dump(out, open('frontend/public/data.json', 'w'))
print("done: frontend/public/data.json")
