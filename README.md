# AI Risk Manager — Fraud-Spike Detector

**Razorpay Buildathon · Track 02: AI Risk Manager**
*"Stop the merchant losing money to fraud, returns and chargebacks"*
#LIVE LINK https://fraud-risk-manager.srisowbarniga2311.workers.dev/
## What this does

A working fraud detector trained on real transaction data, evaluated with
**honest precision/recall metrics on a held-out test set**, and — the key
differentiator — a **false-positive cost analysis** that finds the decision
threshold that actually minimizes business cost, not just the one that
maximizes accuracy.

This is strictly a **defensive detector** — it flags risk, it does not take
any offensive/automated retaliatory action.

## The problem with naive fraud detection

Fraud is rare (0.17% of transactions in this dataset). A model can get
99.8% "accuracy" by just predicting "not fraud" every time — and be
completely useless. So this project focuses on:

1. **Precision & Recall**, not accuracy (accuracy is meaningless on
   imbalanced data)
2. **The real cost of false positives** — every genuine customer wrongly
   blocked is a support ticket and a trust hit, not a free action
3. **The real cost of false negatives** — every missed fraud case is money
   the merchant loses outright
4. **Threshold tuning as a business decision**, not just a model
   hyperparameter

## Dataset

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
— 284,807 transactions, 492 fraud cases (0.173%). Features V1–V28 are
PCA-anonymized; `Amount` and `Time` are raw.

## Approach

1. **Stratified train/test split** (70/30) to preserve the fraud ratio in
   both sets
2. **Random Forest Classifier** with `class_weight='balanced'` to correct
   for the extreme class imbalance
3. **Evaluate at default threshold (0.5)**:
   - Precision: 86.8%
   - Recall: 75.7%
   - ROC-AUC: 0.964
4. **Business cost model**:
   - Cost per false positive ≈ $5 (support cost + customer friction)
   - Cost per false negative ≈ $122.21 (average fraud transaction amount)
5. **Sweep thresholds from 0.05 to 0.90**, compute total cost at each,
   and pick the threshold that minimizes it — not the one with the
   highest F1 score. This is the difference between an ML exercise and
   a Risk Manager an actual merchant would deploy.

## Results

| Threshold | Precision | Recall | Business Cost |
|-----------|-----------|--------|----------------|
| 0.50 (default) | 87% | 76% | $4,485 |
| **0.15 (optimal)** | 48% | **84%** | **$3,608** |

At the optimal threshold, the model **catches more fraud** (84% vs 76%
recall) at **lower total cost** ($3,608 vs $4,485) — because in this
dataset, missing fraud is far more expensive than annoying a few extra
genuine customers with a manual review step.

See `results_chart.png` for the confusion matrix and cost-vs-threshold plot.

## How to run

```bash
pip install -r requirements.txt

# 1. Train the model (takes ~1-2 min)
python train_model.py

# 2. Launch the interactive demo
streamlit run app.py
```

The Streamlit app lets you drag the threshold slider and false-positive
cost input live, and watch precision, recall, and total business cost
update in real time — this is the "recovery workflow" decision layer a
Risk Manager agent would use in production.

## Files

- `train_model.py` — data loading, training, evaluation, threshold sweep
- `app.py` — interactive Streamlit demo
- `fraud_model.pkl`, `scaler.pkl` — saved trained model
- `threshold_analysis.csv` — full cost table across thresholds
- `results_chart.png` — confusion matrix + cost curve visualization

## Honest limitations (worth stating up front)

- Cost figures ($5 friction cost, fraud amount as loss) are illustrative
  assumptions for this demo — a production system would need real
  merchant-specific cost data
- This dataset's V1–V28 features are already anonymized/PCA-transformed,
  so no feature engineering was needed or possible — a live system would
  need real feature engineering from raw transaction metadata
- No SMOTE/oversampling was used — `class_weight='balanced'` was
  sufficient here and avoids synthetic data risks
