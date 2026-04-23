import streamlit as st
import altair as alt
import pandas as pd
import duckdb
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Minimal CSS: bigger fonts only, no theme override ─────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-size: 16px;
    }

    /* Slightly larger axis/label text in charts */
    .vega-embed text { font-size: 13px !important; }

    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #5b6ef5;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0 1rem 0;
        color: #374151;
        font-size: 0.97rem;
        line-height: 1.7;
    }
    .insight-box strong { color: #1e293b; }

    /* Larger metric values */
    [data-testid="stMetricValue"] { font-size: 1.7rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.9rem !important; }

    /* Tab font size */
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem !important; }

    h1 { font-size: 2.1rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.4rem !important; }
    h3 { font-size: 1.15rem !important; }
    p, li { font-size: 1rem; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ── Palette & Altair theme ────────────────────────────────────
PALETTE  = ['#5b6ef5', '#0ea5e9', '#10b981', '#f97316', '#ef4444']

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = "data"
    orders    = pd.read_csv(f"{base}/olist_orders_dataset.csv")
    payments  = pd.read_csv(f"{base}/olist_order_payments_dataset.csv")
    items     = pd.read_csv(f"{base}/olist_order_items_dataset.csv")
    customers = pd.read_csv(f"{base}/olist_customers_dataset.csv")
    reviews   = pd.read_csv(f"{base}/olist_order_reviews_dataset.csv")

    for col in ['order_purchase_timestamp','order_approved_at',
                'order_delivered_carrier_date','order_delivered_customer_date',
                'order_estimated_delivery_date']:
        orders[col] = pd.to_datetime(orders[col], errors='coerce')

    return orders, payments, items, customers, reviews

orders, payments, items, customers, reviews = load_data()

con = duckdb.connect()
con.register('orders_tbl',    orders)
con.register('payments_tbl',  payments)
con.register('items_tbl',     items)
con.register('customers_tbl', customers)
con.register('reviews_tbl',   reviews)


# ── Pre-compute KPIs ──────────────────────────────────────────
total_revenue = payments['payment_value'].sum()
avg_review    = reviews['review_score'].mean()
delivered_df  = orders[orders['order_status'] == 'delivered'].dropna(
    subset=['order_delivered_customer_date','order_estimated_delivery_date'])
late_count    = (delivered_df['order_delivered_customer_date'] >
                 delivered_df['order_estimated_delivery_date']).sum()
late_pct_kpi  = round(100 * late_count / len(delivered_df), 1)

cust_kpi = con.execute("""
    SELECT customer_unique_id, COUNT(order_id) AS total_orders
    FROM orders_tbl o JOIN customers_tbl c ON o.customer_id = c.customer_id
    GROUP BY customer_unique_id
""").df()
returning_rate = round(100 * (cust_kpi['total_orders'] > 1).sum() / len(cust_kpi), 1)


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
with st.container(horizontal_alignment="center"):
    st.title("📦 Olist E-Commerce Analytics")
    st.markdown(
        "An analysis of the Brazilian Olist marketplace covering **payment behavior**, "
        "**delivery performance**, and **customer retention** — "
        "using data from 2016 to 2018."
    )

st.space()

# ── KPI row ───────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Orders",       f"{len(orders):,}")
k2.metric("Total Revenue",      f"R$ {total_revenue:,.0f}")
k3.metric("Avg Review Score",   f"{avg_review:.2f} / 5.00")
k4.metric("Late Delivery Rate", f"{late_pct_kpi}%")
k5.metric("Retention Rate",     f"{returning_rate}%")

st.space("large")

# ── Tabs ──────────────────────────────────────────────────────
tab_eda, tab_pay, tab_del, tab_ret = st.tabs([
    "📈  EDA", "💳  Payment Behavior", "🚚  Delivery & Satisfaction", "🔁  Customer Retention"
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ═══════════════════════════════════════════════════════════════
with tab_eda:
    st.space()

    # ── Monthly trend ─────────────────────────────────────────
    st.subheader("Order Volume Over Time")
    st.caption("Use the slider to zoom into any time window.")

    orders_cp              = orders.copy()
    orders_cp['month']     = orders_cp['order_purchase_timestamp'].dt.to_period('M')
    monthly                = orders_cp.groupby('month').size().reset_index(name='order_count')
    monthly['month_str']   = monthly['month'].astype(str)
    monthly['month_dt']    = pd.to_datetime(monthly['month_str'])

    min_d = monthly['month_dt'].min().to_pydatetime()
    max_d = monthly['month_dt'].max().to_pydatetime()

    date_range = st.slider("Date range", min_value=min_d, max_value=max_d,
                           value=(min_d, max_d), format="MMM YYYY")
    mf = monthly[(monthly['month_dt'] >= date_range[0]) & (monthly['month_dt'] <= date_range[1])]

    area = (
        alt.Chart(mf, title="Monthly Order Volume")
        .mark_area(
            line={"color": PALETTE[0], "strokeWidth": 2.5},
            color=alt.Gradient(
                gradient="linear",
                stops=[alt.GradientStop(color=PALETTE[0], offset=0),
                       alt.GradientStop(color="#e0e7ff", offset=1)],
                x1=1, x2=1, y1=1, y2=0,
            )
        )
        .encode(
            alt.X("month_dt:T", title="Month", axis=alt.Axis(format="%b %Y", labelAngle=-40, labelFontSize=13)),
            alt.Y("order_count:Q", title="Orders", axis=alt.Axis(format=",d", labelFontSize=13)),
            tooltip=[alt.Tooltip("month_str:N", title="Month"),
                     alt.Tooltip("order_count:Q", title="Orders", format=",")]
        )
        .properties(height=320)
    )
    st.altair_chart(area, use_container_width=True)
    insight("Order volume grew steadily through 2017–2018, with a notable <strong>Black Friday spike in Nov 2017</strong>.")

    st.space()

    # ── YoY ───────────────────────────────────────────────────
    st.subheader("Year-over-Year Comparison (Jan – Aug)")
    st.caption("2018 data is capped at August as the dataset ends mid-year.")

    yoy = orders_cp[orders_cp['order_purchase_timestamp'].dt.year.isin([2017, 2018])].copy()
    yoy['year']       = yoy['order_purchase_timestamp'].dt.year.astype(str)
    yoy['month_num']  = yoy['order_purchase_timestamp'].dt.month
    yoy['month_name'] = yoy['order_purchase_timestamp'].dt.strftime('%b')
    yoy = yoy[yoy['month_num'].between(1, 8)]
    yoy_agg    = yoy.groupby(['year','month_num','month_name']).size().reset_index(name='order_count')
    month_sort = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug']

    yoy_chart = (
        alt.Chart(yoy_agg, title="Monthly Orders: 2017 vs 2018")
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            alt.X("month_name:N", title=None, sort=month_sort,
                  axis=alt.Axis(labelAngle=0, labelFontSize=13)),
            alt.Y("order_count:Q", title="Orders", axis=alt.Axis(format=",d", labelFontSize=13)),
            alt.Color("year:N", title="Year",
                      scale=alt.Scale(domain=['2017','2018'], range=[PALETTE[0], PALETTE[2]])),
            alt.XOffset("year:N"),
            tooltip=[alt.Tooltip("month_name:N", title="Month"),
                     alt.Tooltip("year:N", title="Year"),
                     alt.Tooltip("order_count:Q", title="Orders", format=",")]
        )
        .properties(height=320)
    )
    st.altair_chart(yoy_chart, use_container_width=True)
    insight("2018 <strong>consistently outperformed 2017</strong> across every comparable month, with growth accelerating into mid-year.")

    st.space()

    # ── Order status ──────────────────────────────────────────
    st.subheader("Order Status Breakdown")

    status_df = orders['order_status'].value_counts().reset_index()
    status_df.columns = ['Status', 'Count']
    status_df['% of Total'] = (status_df['Count'] / status_df['Count'].sum() * 100).round(1)

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader("Distribution by status")
        status_chart = (
            alt.Chart(status_df, title=alt.TitleParams("", anchor="start"))
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                alt.X("Status:N", sort="-y", axis=alt.Axis(labelAngle=-30, labelFontSize=13)),
                alt.Y("Count:Q", axis=alt.Axis(format=",d", labelFontSize=13), title="Order Count"),
                alt.Color("Status:N", scale=alt.Scale(range=PALETTE), legend=None),
                tooltip=["Status:N",
                         alt.Tooltip("Count:Q", format=","),
                         alt.Tooltip("% of Total:Q", format=".1f", title="% of Total")]
            )
            .properties(height=280)
        )
        st.altair_chart(status_chart, use_container_width=True)

    with cols[1]:
        st.subheader("Summary table")
        st.dataframe(
            status_df,
            use_container_width=True, hide_index=True, height=280,
            column_config={
                "Count":      st.column_config.NumberColumn(format="localized"),
                "% of Total": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            }
        )


# ═══════════════════════════════════════════════════════════════
# TAB 2 — PAYMENT BEHAVIOR
# ═══════════════════════════════════════════════════════════════
with tab_pay:
    st.space()
    st.subheader("Payment Method Overview")
    st.caption("Which payment methods do customers prefer, and how does order value differ across them?")

    pay_sum = con.execute("""
        SELECT
            payment_type,
            COUNT(DISTINCT order_id)                          AS order_count,
            ROUND(AVG(payment_value), 2)                      AS avg_order_value,
            ROUND(SUM(payment_value), 2)                      AS total_revenue,
            ROUND(100.0 * COUNT(DISTINCT order_id)
                  / SUM(COUNT(DISTINCT order_id)) OVER (), 1) AS pct_of_orders
        FROM payments_tbl
        GROUP BY payment_type
        ORDER BY order_count DESC
    """).df()

    metric_choice = st.segmented_control(
        "Visualise by",
        ["% of Orders", "Avg Order Value (BRL)", "Total Revenue (BRL)"],
        default="% of Orders",
    )
    col_map = {
        "% of Orders":           "pct_of_orders",
        "Avg Order Value (BRL)": "avg_order_value",
        "Total Revenue (BRL)":   "total_revenue",
    }
    mcol = col_map[metric_choice] if metric_choice else "pct_of_orders"

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader(f"By {metric_choice}")
        pay_chart = (
            alt.Chart(pay_sum, title=alt.TitleParams("", anchor="start"))
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                alt.Y("payment_type:N", title=None,
                      sort=alt.EncodingSortField(field=mcol, order="descending"),
                      axis=alt.Axis(labelFontSize=13)),
                alt.X(f"{mcol}:Q", title=metric_choice, axis=alt.Axis(labelFontSize=13)),
                alt.Color("payment_type:N", scale=alt.Scale(range=PALETTE), legend=None),
                tooltip=[alt.Tooltip("payment_type:N", title="Method"),
                         alt.Tooltip(f"{mcol}:Q", title=metric_choice, format=",.2f"),
                         alt.Tooltip("order_count:Q", title="Orders", format=",")]
            )
            .properties(height=240)
        )
        st.altair_chart(pay_chart, use_container_width=True)

    with cols[1]:
        st.subheader("Summary table")
        st.dataframe(
            pay_sum.rename(columns={
                'payment_type':'Method','order_count':'Orders',
                'avg_order_value':'Avg Value (R$)','total_revenue':'Revenue (R$)',
                'pct_of_orders':'% Share'
            }),
            use_container_width=True, hide_index=True, height=240,
            column_config={
                "Orders":        st.column_config.NumberColumn(format="localized"),
                "Avg Value (R$)":st.column_config.NumberColumn(format="R$ %.2f"),
                "Revenue (R$)":  st.column_config.NumberColumn(format="R$ %.0f"),
                "% Share":       st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            }
        )

    insight("Credit card dominates with ~74% of all orders and the highest average order value. Boleto is second but skews toward smaller, cost-sensitive purchases.")

    st.space()

    # ── Installments ──────────────────────────────────────────
    st.subheader("Credit Card Installment Behavior")
    st.caption("Bars show order count; the dashed line shows average order value per installment tier.")

    installments = con.execute("""
        SELECT payment_installments,
               COUNT(*) AS order_count,
               ROUND(AVG(payment_value), 2) AS avg_value
        FROM payments_tbl
        WHERE payment_type = 'credit_card'
          AND payment_installments BETWEEN 1 AND 12
        GROUP BY payment_installments
        ORDER BY payment_installments
    """).df()

    inst_range    = st.slider("Installment range", 1, 12, (1, 12))
    inst_filtered = installments[installments['payment_installments'].between(inst_range[0], inst_range[1])]

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader("Orders vs Avg Value by installment")
        inst_bars = (
            alt.Chart(inst_filtered, title=alt.TitleParams("", anchor="start"))
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=PALETTE[0])
            .encode(
                alt.X("payment_installments:O", title="Installments",
                      axis=alt.Axis(labelAngle=0, labelFontSize=13)),
                alt.Y("order_count:Q", title="Number of Orders",
                      axis=alt.Axis(format=",d", labelFontSize=13)),
                tooltip=[alt.Tooltip("payment_installments:O", title="Installments"),
                         alt.Tooltip("order_count:Q", title="Orders", format=","),
                         alt.Tooltip("avg_value:Q", title="Avg Value (R$)", format=",.2f")]
            )
            .properties(height=300)
        )
        avg_line = (
            alt.Chart(inst_filtered)
            .mark_line(color=PALETTE[3], strokeDash=[5,3], strokeWidth=2.5,
                       point=alt.OverlayMarkDef(color=PALETTE[3], filled=True, size=60))
            .encode(
                alt.X("payment_installments:O"),
                alt.Y("avg_value:Q", title="Avg Order Value (R$)",
                      axis=alt.Axis(labelFontSize=13)),
                tooltip=[alt.Tooltip("payment_installments:O", title="Installments"),
                         alt.Tooltip("avg_value:Q", title="Avg Value (R$)", format=",.2f")]
            )
        )
        st.altair_chart(
            alt.layer(inst_bars, avg_line).resolve_scale(y='independent'),
            use_container_width=True
        )

    with cols[1]:
        st.subheader("Summary table")
        st.dataframe(
            inst_filtered.rename(columns={
                'payment_installments':'Installments',
                'order_count':'Orders',
                'avg_value':'Avg Value (R$)'
            }),
            use_container_width=True, hide_index=True, height=300,
            column_config={
                "Orders":        st.column_config.NumberColumn(format="localized"),
                "Avg Value (R$)":st.column_config.NumberColumn(format="R$ %.2f"),
            }
        )

    insight("Most credit card users pay in a <strong>single installment</strong>. A significant portion split across 2–5 installments — a key feature for higher-value purchases.")


# ═══════════════════════════════════════════════════════════════
# TAB 3 — DELIVERY & SATISFACTION
# ═══════════════════════════════════════════════════════════════
with tab_del:
    st.space()
    st.subheader("Delivery Performance vs Review Scores")
    st.caption("Do late deliveries hurt review scores? By how much?")

    delivery_df = con.execute("""
        SELECT
            o.order_id,
            CASE
                WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
                THEN 'Late' ELSE 'On Time'
            END AS delivery_status,
            DATEDIFF('day',
                o.order_estimated_delivery_date,
                o.order_delivered_customer_date
            ) AS days_late,
            r.review_score
        FROM orders_tbl o
        JOIN reviews_tbl r ON o.order_id = r.order_id
        WHERE o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
          AND o.order_estimated_delivery_date IS NOT NULL
    """).df()

    del_sum  = delivery_df.groupby('delivery_status').agg(
        order_count=('order_id','count'), avg_review_score=('review_score','mean')
    ).round(2).reset_index()
    late_pct = round(100*(delivery_df['delivery_status']=='Late').sum()/len(delivery_df), 1)
    on_score = del_sum[del_sum['delivery_status']=='On Time']['avg_review_score'].values[0]
    lt_score = del_sum[del_sum['delivery_status']=='Late']['avg_review_score'].values[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Late Deliveries",     f"{late_pct}%")
    m2.metric("Avg Score — On Time", f"{on_score:.2f} ★")
    m3.metric("Avg Score — Late",    f"{lt_score:.2f} ★",
              delta=f"{lt_score-on_score:.2f}", delta_color="inverse")

    st.space()

    status_filter = st.multiselect(
        "Filter by delivery status",
        options=["On Time","Late"], default=["On Time","Late"]
    )
    fd = delivery_df[delivery_df['delivery_status'].isin(status_filter)]
    score_dist = fd.groupby(['delivery_status','review_score']).size().reset_index(name='count')
    score_dist['pct'] = score_dist.groupby('delivery_status')['count'].transform(
        lambda x: x / x.sum() * 100
    )

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader("Avg review score")
        bar_sum = (
            alt.Chart(del_sum[del_sum['delivery_status'].isin(status_filter)],
                      title=alt.TitleParams("", anchor="start"))
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                alt.X("delivery_status:N", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=14)),
                alt.Y("avg_review_score:Q", title="Avg Review Score",
                      scale=alt.Scale(domain=[0, 5.5]),
                      axis=alt.Axis(labelFontSize=13)),
                alt.Color("delivery_status:N",
                          scale=alt.Scale(domain=["On Time","Late"],
                                          range=[PALETTE[2], PALETTE[4]]), legend=None),
                tooltip=[alt.Tooltip("delivery_status:N", title="Status"),
                         alt.Tooltip("avg_review_score:Q", title="Avg Score", format=".2f"),
                         alt.Tooltip("order_count:Q", title="Orders", format=",")]
            )
            .properties(height=300)
        )
        labels = bar_sum.mark_text(dy=-12, fontSize=16, fontWeight=700).encode(
            text=alt.Text("avg_review_score:Q", format=".2f"),
            color=alt.value("#1e293b")
        )
        st.altair_chart(bar_sum + labels, use_container_width=True)

    with cols[1]:
        st.subheader("Score distribution")
        line_dist = (
            alt.Chart(score_dist, title=alt.TitleParams("", anchor="start"))
            .mark_line(point=alt.OverlayMarkDef(filled=True, size=100), strokeWidth=2.5)
            .encode(
                alt.X("review_score:O", title="Review Score",
                      axis=alt.Axis(labelAngle=0, labelFontSize=13)),
                alt.Y("pct:Q", title="% of Orders", axis=alt.Axis(format=".1f", labelFontSize=13)),
                alt.Color("delivery_status:N",
                          scale=alt.Scale(domain=["On Time","Late"],
                                          range=[PALETTE[2], PALETTE[4]]), title="Status"),
                tooltip=[alt.Tooltip("delivery_status:N", title="Status"),
                         alt.Tooltip("review_score:O", title="Score"),
                         alt.Tooltip("pct:Q", title="% of Orders", format=".1f"),
                         alt.Tooltip("count:Q", title="Count", format=",")]
            )
            .properties(height=300)
        )
        st.altair_chart(line_dist, use_container_width=True)

    insight("Late deliveries have a <strong>dramatically lower average review score</strong>. Late orders spike sharply at 1-star, while on-time orders skew heavily toward 5-stars.")

    st.space()

    # ── Lateness buckets ──────────────────────────────────────
    st.subheader("Score vs Lateness Severity")
    st.caption("The longer the delay, the steeper the satisfaction drop.")

    dls = con.execute("""
        SELECT
            CASE
                WHEN days_late <= 0  THEN '0 – On Time'
                WHEN days_late <= 3  THEN '1–3 days late'
                WHEN days_late <= 7  THEN '4–7 days late'
                WHEN days_late <= 14 THEN '8–14 days late'
                ELSE '15+ days late'
            END AS lateness_bucket,
            COUNT(*) AS order_count,
            ROUND(AVG(review_score), 2) AS avg_score
        FROM (
            SELECT r.review_score,
                   DATEDIFF('day', o.order_estimated_delivery_date,
                       o.order_delivered_customer_date) AS days_late
            FROM orders_tbl o
            JOIN reviews_tbl r ON o.order_id = r.order_id
            WHERE o.order_status = 'delivered'
              AND o.order_delivered_customer_date IS NOT NULL
              AND o.order_estimated_delivery_date IS NOT NULL
        )
        GROUP BY lateness_bucket
    """).df()

    bucket_order = ['0 – On Time','1–3 days late','4–7 days late','8–14 days late','15+ days late']
    dls['lateness_bucket'] = pd.Categorical(dls['lateness_bucket'], categories=bucket_order, ordered=True)
    dls = dls.sort_values('lateness_bucket')
    dls['color'] = dls['lateness_bucket'].apply(lambda x: PALETTE[2] if x=='0 – On Time' else PALETTE[4])

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader("Avg score by lateness bucket")
        bucket_chart = (
            alt.Chart(dls, title=alt.TitleParams("", anchor="start"))
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                alt.X("lateness_bucket:N", title=None, sort=bucket_order,
                      axis=alt.Axis(labelAngle=-20, labelFontSize=12)),
                alt.Y("avg_score:Q", title="Avg Review Score",
                      scale=alt.Scale(domain=[0, 5.5]),
                      axis=alt.Axis(labelFontSize=13)),
                alt.Color("color:N", scale=None, legend=None),
                tooltip=[alt.Tooltip("lateness_bucket:N", title="Lateness"),
                         alt.Tooltip("avg_score:Q", title="Avg Score", format=".2f"),
                         alt.Tooltip("order_count:Q", title="Orders", format=",")]
            )
            .properties(height=300)
        )
        labels = bucket_chart.mark_text(dy=-12, fontSize=14, fontWeight=700).encode(
            text=alt.Text("avg_score:Q", format=".2f"),
            color=alt.value("#1e293b")
        )
        st.altair_chart(bucket_chart + labels, use_container_width=True)

    with cols[1]:
        st.subheader("Summary table")
        st.dataframe(
            dls[['lateness_bucket','order_count','avg_score']].rename(columns={
                'lateness_bucket':'Lateness','order_count':'Orders','avg_score':'Avg Score'
            }),
            use_container_width=True, hide_index=True, height=300,
            column_config={
                "Orders":    st.column_config.NumberColumn(format="localized"),
                "Avg Score": st.column_config.ProgressColumn(min_value=0, max_value=5, format="%.2f"),
            }
        )

    insight("Score decline is <strong>progressive and steep</strong> — orders 15+ days late drop close to 2 stars on average, versus 4.2 for on-time deliveries.")


# ═══════════════════════════════════════════════════════════════
# TAB 4 — CUSTOMER RETENTION
# ═══════════════════════════════════════════════════════════════
with tab_ret:
    st.space()
    st.subheader("Customer Retention")
    st.caption("What share of customers return for a second order, and how long does it take?")

    cust_orders = con.execute("""
        SELECT c.customer_unique_id, COUNT(o.order_id) AS total_orders
        FROM orders_tbl o JOIN customers_tbl c ON o.customer_id = c.customer_id
        GROUP BY c.customer_unique_id
    """).df()

    one_time  = int((cust_orders['total_orders'] == 1).sum())
    returning = int((cust_orders['total_orders'] > 1).sum())
    total     = len(cust_orders)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Unique Customers", f"{total:,}")
    m2.metric("One-time Buyers",        f"{one_time:,}", delta=f"{100*one_time/total:.1f}% of base")
    m3.metric("Returning Buyers",       f"{returning:,}", delta=f"{100*returning/total:.1f}% of base")

    st.space()

    repeat_gap = con.execute("""
        WITH ranked AS (
            SELECT c.customer_unique_id, o.order_purchase_timestamp,
                   ROW_NUMBER() OVER (PARTITION BY c.customer_unique_id
                                      ORDER BY o.order_purchase_timestamp) AS order_rank
            FROM orders_tbl o JOIN customers_tbl c ON o.customer_id = c.customer_id
        ),
        first_second AS (
            SELECT r1.customer_unique_id,
                   DATEDIFF('day', r1.order_purchase_timestamp, r2.order_purchase_timestamp) AS days_to_return
            FROM ranked r1
            JOIN ranked r2
              ON r1.customer_unique_id = r2.customer_unique_id
             AND r1.order_rank = 1 AND r2.order_rank = 2
        )
        SELECT * FROM first_second WHERE days_to_return > 0
    """).df()

    median_days = int(repeat_gap['days_to_return'].median())
    mean_days   = int(repeat_gap['days_to_return'].mean())

    freq_df = pd.DataFrame({
        'Segment': ['One-time','Returning'],
        'Count':   [one_time, returning],
        'pct':     [round(100*one_time/total,1), round(100*returning/total,1)]
    })

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader("Purchase frequency breakdown")
        donut = (
            alt.Chart(freq_df, title=alt.TitleParams("", anchor="start"))
            .mark_arc(innerRadius=80, outerRadius=150)
            .encode(
                alt.Theta("Count:Q"),
                alt.Color("Segment:N",
                          scale=alt.Scale(domain=['One-time','Returning'],
                                          range=[PALETTE[1], PALETTE[0]])),
                tooltip=[alt.Tooltip("Segment:N", title="Segment"),
                         alt.Tooltip("Count:Q", title="Customers", format=","),
                         alt.Tooltip("pct:Q", title="% of Base", format=".1f")]
            )
            .properties(height=340)
        )
        st.altair_chart(donut, use_container_width=True)

    with cols[1]:
        st.subheader("Days until customers return")
        clip_val = st.slider("Clip at (days)", 90, 730, 365, step=30,
                             help="Hides long-tail outliers to focus on the typical return window")
        clipped  = repeat_gap[repeat_gap['days_to_return'] <= clip_val].copy()

        hist = (
            alt.Chart(clipped, title=alt.TitleParams("", anchor="start"))
            .mark_bar(color=PALETTE[0], opacity=0.8, binSpacing=1)
            .encode(
                alt.X("days_to_return:Q", title="Days Between 1st and 2nd Order",
                      bin=alt.Bin(maxbins=40), axis=alt.Axis(labelFontSize=13)),
                alt.Y("count():Q", title="Customers", axis=alt.Axis(format=",d", labelFontSize=13)),
                tooltip=[alt.Tooltip("days_to_return:Q", title="Days (bin)", bin="binned"),
                         alt.Tooltip("count():Q", title="Customers", format=",")]
            )
            .properties(height=280)
        )
        med_rule = (
            alt.Chart(pd.DataFrame({'median': [median_days]}))
            .mark_rule(color=PALETTE[3], strokeDash=[6,3], strokeWidth=2.5)
            .encode(x="median:Q")
        )
        st.altair_chart(hist + med_rule, use_container_width=True)
        st.caption(
            f"Median return time: **{median_days} days** · "
            f"Mean: **{mean_days} days** · "
            f"Sample: **{len(clipped):,}** returning customers"
        )

    insight(
        f"<strong>{100*one_time/total:.0f}%</strong> of customers only purchase once. "
        f"Among those who return, the median gap is <strong>{median_days} days</strong>. "
        "Loyalty cashback or incentives targeting first-time buyers within 30–60 days "
        "could meaningfully improve second-purchase conversion."
    )