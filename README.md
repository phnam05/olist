# 📦 Olist E-Commerce Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://phnam05-olist.streamlit.app/)

An interactive Streamlit dashboard for analyzing the Brazilian **Olist e-commerce dataset (2016–2018)**, covering payment behavior, delivery performance, customer retention, RFM segmentation, and churn prediction.

---

## Features

### 📈 EDA
- Monthly order trends with an interactive date range slider
- Year-over-year comparison (2017 vs. 2018)
- Order status breakdown with summary table

### 💳 Payment Behavior
- Payment method distribution by order count, average value, and total revenue
- Credit card installment analysis — order volume and average spend per installment tier

### 🚚 Delivery & Satisfaction
- Late vs. on-time delivery impact on review scores
- Lateness severity bucketing (1–3 days → 15+ days) with average score per bucket
- **Mann-Whitney U test** confirming the score gap is statistically significant (p < 0.001, effect size r ≈ 0.3)

### 🔁 Customer Retention
- One-time vs. returning customer breakdown
- Distribution of days between first and second purchase
- Cumulative return curve with 30 / 90 / 180-day milestones

### 🎯 RFM Segmentation
- Quintile-based Recency, Frequency, and Monetary scoring
- Six named segments: Champions, Loyal Customers, New Customers, Potential Loyalists, At Risk, Lost
- Scatter plot of customer map colored by segment

### 🤖 Churn Prediction
- Binary classification: will a customer make a second purchase?
- Four models compared: Logistic Regression and Gradient Boosting × two imbalance strategies (class weighting and undersampling)
- Metrics: ROC-AUC, Precision, Recall, F1 on the retained class
- Interactive live predictor — adjust order features and get a real-time churn probability estimate

---

## Tech Stack

| Layer | Tools |
|---|---|
| App framework | Streamlit |
| Data manipulation | Pandas, DuckDB |
| Visualization | Altair |
| Machine learning | Scikit-learn |
| Statistics | SciPy |

---

## Project Structure

```
.
├── app.py
├── requirements.txt
├── data/
│   ├── olist_orders_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_customers_dataset.csv
│   └── olist_order_reviews_dataset.csv
└── README.md
```

---

## Getting Started

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py
```

---

## Data Source

[Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — real-world transactional data covering orders, payments, customers, and reviews from 2016 to 2018.

---

## Key Findings

- Order volume grew ~50× from late 2016 to mid-2018, with a clear Black Friday spike in November 2017
- Credit cards account for ~74% of transactions and carry the highest average order value
- Late deliveries drop average review scores by ~1.5 points (4.2 → 2.7), with the effect confirmed statistically
- Over 95% of customers never make a second purchase — the median return window for those who do is ~150 days
- Review score at first purchase is the strongest single predictor of churn

---

## Potential Extensions

- Cohort analysis by acquisition month
- Seller-side analytics (top sellers, fulfillment rates)
- Geographic heatmap by state
- Additional ML features (product category, seller region) to push AUC above 0.70
