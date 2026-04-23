# 📦 Olist E-Commerce Analytics (Streamlit App)

## View it in one click with Streamlit
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://phnam05-olist.streamlit.app/)

## Overview

This project is an interactive **Streamlit dashboard** for analyzing the
Brazilian **Olist e-commerce dataset (2016--2018)**.

It explores: - 📈 Order trends over time - 💳 Payment behavior and
installment usage - 🚚 Delivery performance and its impact on customer
satisfaction - 🔁 Customer retention and repeat purchase behavior

The app is designed for **exploratory data analysis (EDA)** and business
insight generation.

------------------------------------------------------------------------

## Features

### 🔢 Key Metrics

-   Total Orders
-   Total Revenue
-   Average Review Score
-   Late Delivery Rate
-   Customer Retention Rate

### 📈 EDA

-   Monthly order trends with time filtering
-   Year-over-year comparison (2017 vs 2018)
-   Order status breakdown

### 💳 Payment Behavior

-   Payment method distribution
-   Average order value by method
-   Credit card installment analysis

### 🚚 Delivery & Satisfaction

-   Late vs on-time delivery comparison
-   Impact of delays on review scores
-   Lateness severity analysis

### 🔁 Customer Retention

-   One-time vs returning customers
-   Time between first and second purchase
-   Retention distribution visualization

------------------------------------------------------------------------

## Tech Stack

-   **Python**
-   **Streamlit**
-   **Pandas**
-   **DuckDB** (for SQL-based analysis)
-   **Altair** (for interactive visualizations)

------------------------------------------------------------------------

## Project Structure

    .
    ├── app.py
    ├── data/
    │   ├── olist_orders_dataset.csv
    │   ├── olist_order_payments_dataset.csv
    │   ├── olist_order_items_dataset.csv
    │   ├── olist_customers_dataset.csv
    │   └── olist_order_reviews_dataset.csv
    └── README.md

------------------------------------------------------------------------

## Installation

1.  Clone the repository:

``` bash
git clone <your-repo-url>
cd <your-repo>
```

2.  Install dependencies:

``` bash
pip install -r requirements.txt
```

3.  Run the app:

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## Data Source

Dataset: **Olist Brazilian E-Commerce Public Dataset**

Contains real-world transactional data including: - Orders - Payments -
Customers - Reviews

------------------------------------------------------------------------

## Key Insights

-   📊 Order volume grew significantly through 2017--2018
-   💳 Credit cards dominate (\~74% of transactions)
-   🚚 Late deliveries significantly reduce customer satisfaction
-   ⭐ Severe delays (15+ days) drop ratings close to 2 stars
-   🔁 Majority of customers are one-time buyers
-   ⏳ Median return time \~ months → opportunity for retention
    campaigns

------------------------------------------------------------------------

## Future Improvements

-   Add cohort analysis
-   Customer segmentation (RFM)
-   Predictive modeling (churn / LTV)
-   Deployment (Streamlit Cloud)

------------------------------------------------------------------------

## Author

Your Name

------------------------------------------------------------------------

## License

MIT License
