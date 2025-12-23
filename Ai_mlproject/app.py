import streamlit as st
import pandas as pd
import joblib
import os

import plotly.graph_objects as go
from collections import Counter

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ReviewSense",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS (UI TOKOPEDIA / SHOPEE)
# =========================
st.markdown("""
<style>
body {
    background-color: #f5f6fa;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.metric-title {
    font-size: 14px;
    color: gray;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
}
.section-title {
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR BRANDING
# =========================
st.sidebar.markdown("""
<div style="text-align:center">
    <h2>📊 ReviewSense</h2>
    <p style="color:gray; font-size:14px">
    AI Review & Reputation Dashboard
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "model", "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "model", "vectorizer.pkl"))

# =========================
# UPLOAD CSV
# =========================
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload CSV Review",
    type=["csv"]
)

if not uploaded_file:
    st.info("⬅️ Upload file CSV untuk memulai analisis")
    st.stop()

df = pd.read_csv(uploaded_file, engine="python")

# =========================
# VALIDASI KOLOM
# =========================
required_cols = ["product_name", "review_text"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Kolom `{col}` wajib ada")
        st.stop()

# =========================
# PREDICT SENTIMENT
# =========================
X = vectorizer.transform(df["review_text"].astype(str))
df["sentiment"] = model.predict(X)

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.markdown("### 🔍 Filter Produk")
products = ["SEMUA PRODUK"] + sorted(df["product_name"].unique())
selected_product = st.sidebar.selectbox("Pilih Produk", products)

if selected_product == "SEMUA PRODUK":
    filtered_df = df.copy()
else:
    filtered_df = df[df["product_name"] == selected_product]

# =========================
# METRICS
# =========================
count = filtered_df["sentiment"].value_counts()
pos = count.get("positive", 0)
neu = count.get("neutral", 0)
neg = count.get("negative", 0)
total = pos + neu + neg
health = round((pos / total) * 100, 1) if total else 0

# =========================
# HEADER
# =========================
st.markdown(f"""
<div class="card">
    <div class="section-title">🛍️ Dashboard Reputasi Produk</div>
    <p style="color:gray">
    Produk aktif: <b>{selected_product}</b>
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# KPI CARDS
# =========================
c1, c2, c3, c4 = st.columns(4)

def metric_card(col, title, value, emoji):
    with col:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">{emoji} {title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

metric_card(c1, "Positive Review", pos, "😊")
metric_card(c2, "Neutral Review", neu, "😐")
metric_card(c3, "Negative Review", neg, "😡")
metric_card(c4, "Health Score", f"{health}%", "❤️")

# =========================
# CHART + INSIGHT
# =========================
left, right = st.columns([2,1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    fig = go.Figure(go.Pie(
        labels=["Positive", "Neutral", "Negative"],
        values=[pos, neu, neg],
        hole=0.6
    ))
    fig.update_layout(
        title="Distribusi Sentimen",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Insight AI")

    neg_reviews = filtered_df[filtered_df["sentiment"] == "negative"]["review_text"]

    if len(neg_reviews) == 0:
        st.success("🎉 Tidak ada masalah signifikan")
    else:
        words = " ".join(neg_reviews).lower().split()
        common = Counter(words).most_common(5)
        issues = ", ".join([w[0] for w in common])
        st.warning(f"Masalah utama: **{issues}**")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# DETAIL REVIEW
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📝 Detail Review")

st.dataframe(
    filtered_df[["product_name", "review_text", "sentiment"]],
    use_container_width=True,
    height=350
)

st.markdown("</div>", unsafe_allow_html=True)
