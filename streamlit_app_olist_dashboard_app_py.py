import streamlit as st
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title='Olist E-commerce Dashboard', layout='wide')

PALETTE = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']

@st.cache_data

def load_data(data_path='data'):
    data_path = Path(data_path)
    orders = pd.read_csv(data_path / 'olist_orders_dataset.csv', parse_dates=[
        'order_purchase_timestamp',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ])
    payments = pd.read_csv(data_path / 'olist_order_payments_dataset.csv')
    customers = pd.read_csv(data_path / 'olist_customers_dataset.csv')
    reviews = pd.read_csv(data_path / 'olist_order_reviews_dataset.csv')
    items = pd.read_csv(data_path / 'olist_order_items_dataset.csv')
    return orders, payments, customers, reviews, items

orders, payments, customers, reviews, items = load_data()

con = duckdb.connect()
con.register('orders_tbl', orders)
con.register('payments_tbl', payments)
con.register('customers_tbl', customers)
con.register('reviews_tbl', reviews)
con.register('items_tbl', items)

st.title('Olist E-commerce Data Analysis Dashboard')
st.markdown('Interactive business insights dashboard built from your notebook analysis')

section = st.sidebar.selectbox(
    'Choose analysis section',
    ['Overview', 'Payment Behavior', 'Delivery & Satisfaction', 'Customer Retention']
)

if section == 'Overview':
    st.header('Monthly Orders Trend')
    monthly = orders.groupby(orders['order_purchase_timestamp'].dt.to_period('M')).size().reset_index(name='order_count')
    monthly['order_purchase_timestamp'] = monthly['order_purchase_timestamp'].astype(str)
    st.line_chart(monthly.set_index('order_purchase_timestamp'))

elif section == 'Payment Behavior':
    st.header('Payment Method Analysis')
    query = """
        SELECT
            payment_type,
            COUNT(DISTINCT order_id) AS order_count,
            ROUND(AVG(payment_value), 2) AS avg_order_value,
            ROUND(SUM(payment_value), 2) AS total_revenue
        FROM payments_tbl
        GROUP BY payment_type
        ORDER BY total_revenue DESC
    """
    payment_summary = con.execute(query).df()
    st.dataframe(payment_summary, use_container_width=True)
    st.bar_chart(payment_summary.set_index('payment_type')['total_revenue'])

elif section == 'Delivery & Satisfaction':
    st.header('Late Delivery vs Review Score')
    query = """
        SELECT
            CASE
                WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 'Late'
                ELSE 'On Time'
            END AS delivery_status,
            ROUND(AVG(r.review_score), 2) AS avg_review_score,
            COUNT(*) AS total_orders
        FROM orders_tbl o
        JOIN reviews_tbl r ON o.order_id = r.order_id
        WHERE o.order_delivered_customer_date IS NOT NULL
        GROUP BY delivery_status
    """
    summary = con.execute(query).df()
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(summary, use_container_width=True)
    with col2:
        st.bar_chart(summary.set_index('delivery_status')['avg_review_score'])

elif section == 'Customer Retention':
    st.header('Repeat Customer Analysis')
    query = """
        SELECT
            c.customer_unique_id,
            COUNT(o.order_id) AS total_orders
        FROM orders_tbl o
        JOIN customers_tbl c ON o.customer_id = c.customer_id
        GROUP BY c.customer_unique_id
    """
    customer_orders = con.execute(query).df()
    returning = (customer_orders['total_orders'] > 1).sum()
    one_time = (customer_orders['total_orders'] == 1).sum()

    st.metric('Returning Customers', returning)
    st.metric('One-time Customers', one_time)

    fig, ax = plt.subplots()
    ax.pie([one_time, returning], labels=['One-time', 'Returning'], autopct='%1.1f%%')
    st.pyplot(fig)
