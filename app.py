import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# ── Konfigurasi halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: #495057 !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #212529 !important;
        font-weight: 700 !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    .churn-box { padding: 20px; border-radius: 12px; text-align: center; font-size: 20px; font-weight: bold; }
    .churn-yes { background-color: #f8d7da; color: #842029; border: 2px solid #f5c2c7; }
    .churn-no  { background-color: #d1e7dd; color: #0f5132; border: 2px solid #badbcc; }

    .rec-box {
        background-color: #cfe2ff;
        color: #084298 !important;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #b6d4fe;
        font-size: 14px;
        min-height: 110px;
    }
    .rec-box b { color: #084298 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Artefak ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model             = joblib.load("best_model.pkl")
    scaler            = joblib.load("scaler.pkl")
    selected_features = joblib.load("selected_features.pkl")
    all_columns       = joblib.load("all_columns.pkl")
    category_options  = joblib.load("category_options.pkl")
    return model, scaler, selected_features, all_columns, category_options

model, scaler, selected_features, all_columns, category_options = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction")
st.markdown(
    "Prediksi apakah seorang pelanggan akan **churn** atau **bertahan** "
    "berdasarkan data profil dan perilaku mereka."
)
st.divider()

# ── Sidebar: Form Input ───────────────────────────────────────────────────────
st.sidebar.header("🧾 Input Data Pelanggan")

def user_input():
    with st.sidebar:

        st.subheader("👤 Demografi")
        gender  = st.selectbox("Gender",  category_options.get("gender",  ["Male", "Female"]))
        age     = st.slider("Usia (Age)", 18, 80, 35)
        country = st.selectbox("Negara",  category_options.get("country", ["USA"]))
        city    = st.selectbox("Kota",    category_options.get("city",    ["New York"]))

        st.subheader("📦 Langganan & Akuisisi")
        acquisition_channel = st.selectbox(
            "Acquisition Channel",
            category_options.get("acquisition_channel", ["Email"])
        )
        device_type = st.selectbox(
            "Device Type",
            category_options.get("device_type", ["Mobile"])
        )
        subscription_type = st.selectbox(
            "Subscription Type",
            category_options.get("subscription_type", ["Basic"])
        )
        is_premium_user = st.radio(
            "Premium User?", [0, 1],
            format_func=lambda x: "Ya" if x == 1 else "Tidak",
            horizontal=True
        )

        st.subheader("🖱️ Aktivitas")
        total_visits      = st.slider("Total Visits",              0,    50,   15)
        avg_session_time  = st.slider("Avg Session Time (menit)",  0.0,  30.0,  8.0, step=0.5)
        pages_per_session = st.slider("Pages per Session",         0.0,  15.0,  4.0, step=0.5)

        st.subheader("📧 Email Engagement")
        email_open_rate  = st.slider("Email Open Rate",  0.0, 1.0, 0.50, step=0.01)
        email_click_rate = st.slider("Email Click Rate", 0.0, 0.5, 0.25, step=0.01)

        st.subheader("💰 Transaksi")
        total_spent     = st.number_input("Total Spent (USD)",      0.0, 20000.0,  500.0, step=50.0)
        avg_order_value = st.slider("Avg Order Value (USD)",        0.0,   200.0,   60.0, step=1.0)
        discount_used   = st.radio(
            "Gunakan Diskon?", [0, 1],
            format_func=lambda x: "Ya" if x == 1 else "Tidak",
            horizontal=True
        )
        payment_method  = st.selectbox(
            "Payment Method",
            category_options.get("payment_method", ["Credit Card"])
        )

        st.subheader("🎧 Layanan & Kepuasan")
        support_tickets      = st.slider("Support Tickets",        0,  10,   2)
        refund_requested     = st.radio(
            "Pernah Refund?", [0, 1],
            format_func=lambda x: "Ya" if x == 1 else "Tidak",
            horizontal=True
        )
        delivery_delay_days  = st.slider("Delivery Delay (hari)",  0,  15,   3)
        satisfaction_score   = st.slider("Satisfaction Score",     1.0, 5.0, 3.5, step=0.5)
        nps_score            = st.slider("NPS Score",              0,  10,   5)

        st.subheader("📈 Marketing & Nilai")
        marketing_spend_per_user   = st.slider("Marketing Spend / User (USD)", 5.0, 30.0, 17.0, step=0.5)
        lifetime_value             = st.number_input("Lifetime Value (USD)",    0.0, 5000.0, 1200.0, step=100.0)
        last_3_month_purchase_freq = st.slider("Purchase Freq (3 bln terakhir)", 0, 14, 7)

    return pd.DataFrame([{
        "gender"                    : gender,
        "age"                       : age,
        "country"                   : country,
        "city"                      : city,
        "acquisition_channel"       : acquisition_channel,
        "device_type"               : device_type,
        "subscription_type"         : subscription_type,
        "is_premium_user"           : is_premium_user,
        "total_visits"              : total_visits,
        "avg_session_time"          : avg_session_time,
        "pages_per_session"         : pages_per_session,
        "email_open_rate"           : email_open_rate,
        "email_click_rate"          : email_click_rate,
        "total_spent"               : total_spent,
        "avg_order_value"           : avg_order_value,
        "discount_used"             : discount_used,
        "support_tickets"           : support_tickets,
        "refund_requested"          : refund_requested,
        "delivery_delay_days"       : delivery_delay_days,
        "payment_method"            : payment_method,
        "satisfaction_score"        : satisfaction_score,
        "nps_score"                 : nps_score,
        "marketing_spend_per_user"  : marketing_spend_per_user,
        "lifetime_value"            : lifetime_value,
        "last_3_month_purchase_freq": last_3_month_purchase_freq,
    }])

input_df = user_input()

# ── Preprocessing Input ───────────────────────────────────────────────────────
def preprocess_input(input_df, all_columns, scaler, selected_features):
    df_in = input_df.copy()

    cat_cols     = df_in.select_dtypes(include="object").columns.tolist()
    binary_cats  = [c for c in cat_cols if df_in[c].nunique() <= 2]
    nominal_cats = [c for c in cat_cols if df_in[c].nunique() > 2]

    le = LabelEncoder()
    for col in binary_cats:
        df_in[col] = le.fit_transform(df_in[col].astype(str))

    df_in = pd.get_dummies(df_in, columns=nominal_cats, drop_first=True)

    # Selaraskan kolom dengan saat training
    for col in all_columns:
        if col not in df_in.columns:
            df_in[col] = 0
    df_in = df_in[all_columns]

    # Scaling → Feature Selection
    df_scaled = pd.DataFrame(scaler.transform(df_in), columns=all_columns)
    return df_scaled[selected_features]

processed = preprocess_input(input_df, all_columns, scaler, selected_features)

# ── Prediksi ──────────────────────────────────────────────────────────────────
prediction    = model.predict(processed)[0]
proba         = model.predict_proba(processed)[0]
churn_prob    = proba[1]
no_churn_prob = proba[0]

# ── Hasil Prediksi ────────────────────────────────────────────────────────────
st.subheader("🎯 Hasil Prediksi")

col_result, col_prob1, col_prob2 = st.columns(3)

with col_result:
    if prediction == 1:
        st.markdown(
            '<div class="churn-box churn-yes">⚠️ Pelanggan Diprediksi CHURN</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="churn-box churn-no">✅ Pelanggan Diprediksi TIDAK CHURN</div>',
            unsafe_allow_html=True
        )

with col_prob1:
    st.metric(
        label="🔴 Probabilitas Churn",
        value=f"{churn_prob*100:.1f}%",
        delta=f"{(churn_prob - 0.5)*100:+.1f}% dari threshold",
        delta_color="inverse"
    )

with col_prob2:
    st.metric(
        label="🟢 Probabilitas Bertahan",
        value=f"{no_churn_prob*100:.1f}%"
    )

st.divider()

# ── Visualisasi ───────────────────────────────────────────────────────────────
st.subheader("📉 Visualisasi Risiko Churn")

col_gauge, col_bar = st.columns(2)

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = round(churn_prob * 100, 1),
        title = {"text": "Churn Risk (%)", "font": {"size": 18}},
        delta = {"reference": 50, "suffix": "%"},
        gauge = {
            "axis" : {"range": [0, 100], "tickwidth": 1},
            "bar"  : {"color": "#dc3545" if churn_prob > 0.5 else "#198754", "thickness": 0.3},
            "steps": [
                {"range": [0,  40], "color": "#d1e7dd"},
                {"range": [40, 60], "color": "#fff3cd"},
                {"range": [60,100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line"     : {"color": "black", "width": 4},
                "thickness": 0.75,
                "value"    : 50
            }
        }
    ))
    fig_gauge.update_layout(height=320, margin=dict(t=60, b=20, l=30, r=30))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_bar:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors  = ["#198754", "#dc3545"]
    bars    = ax.barh(
        ["Tidak Churn", "Churn"],
        [no_churn_prob, churn_prob],
        color=colors, edgecolor="white", height=0.5
    )
    for bar, val in zip(bars, [no_churn_prob, churn_prob]):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val*100:.1f}%", va="center", fontsize=12, fontweight="bold"
        )
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Probabilitas", fontsize=11)
    ax.set_title("Distribusi Probabilitas", fontsize=12, fontweight="bold")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Threshold 50%")
    ax.legend(fontsize=9)
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# ── Rekomendasi Aksi ──────────────────────────────────────────────────────────
st.subheader("💡 Rekomendasi Aksi")

if prediction == 1:
    st.warning("Pelanggan ini berisiko churn. Pertimbangkan langkah berikut:")
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.markdown(
        '<div class="rec-box">🎁 <b>Tawarkan Diskon</b><br>Berikan promo atau voucher eksklusif untuk mempertahankan pelanggan.</div>',
        unsafe_allow_html=True
    )
    col_r2.markdown(
        '<div class="rec-box">📞 <b>Hubungi Langsung</b><br>Lakukan outreach personal via email atau telepon untuk memahami kendala pelanggan.</div>',
        unsafe_allow_html=True
    )
    col_r3.markdown(
        '<div class="rec-box">⭐ <b>Upgrade Premium</b><br>Tawarkan uji coba gratis paket premium untuk meningkatkan engagement.</div>',
        unsafe_allow_html=True
    )
else:
    st.success("Pelanggan ini diprediksi akan bertahan. Pertimbangkan langkah berikut:")
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.markdown(
        '<div class="rec-box">🚀 <b>Tingkatkan Engagement</b><br>Kirim konten relevan dan personalisasi pengalaman pelanggan.</div>',
        unsafe_allow_html=True
    )
    col_r2.markdown(
        '<div class="rec-box">🎯 <b>Program Loyalitas</b><br>Ajak pelanggan bergabung ke program rewards untuk memperkuat loyalitas.</div>',
        unsafe_allow_html=True
    )
    col_r3.markdown(
        '<div class="rec-box">📊 <b>Upselling</b><br>Tawarkan produk atau layanan tambahan yang sesuai dengan profil pelanggan.</div>',
        unsafe_allow_html=True
    )

st.divider()

# ── Detail Input & Fitur ──────────────────────────────────────────────────────
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    with st.expander("📋 Data Input Pelanggan"):
        st.dataframe(
            input_df.T.rename(columns={0: "Nilai"}),
            use_container_width=True
        )

with col_exp2:
    with st.expander("📖 Penjelasan Fitur Kunci"):
        fitur_info = {
            "satisfaction_score"          : "Skor kepuasan (1–5). Makin rendah → risiko churn makin tinggi.",
            "total_spent"                 : "Total pengeluaran. Spending tinggi → lebih loyal.",
            "support_tickets"             : "Jumlah tiket support. Banyak tiket → potensi ketidakpuasan.",
            "nps_score"                   : "Net Promoter Score (0–10). Rendah → pelanggan tidak puas.",
            "lifetime_value"              : "Estimasi nilai total pelanggan selama berlangganan.",
            "last_3_month_purchase_freq"  : "Frekuensi beli 3 bulan terakhir. Rendah → risiko churn naik.",
            "email_open_rate"             : "% email yang dibuka. Rendah → keterlibatan menurun.",
            "is_premium_user"             : "Pengguna premium umumnya lebih sulit churn.",
        }
        for fitur, penjelasan in fitur_info.items():
            st.markdown(f"**`{fitur}`** — {penjelasan}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🎓 UAS Bengkel Koding Data Science — Universitas Dian Nuswantoro  |  "
    "Customer Churn Prediction App  |  2025"
)