import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudSense AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- fonts ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {font-family: 'Inter', sans-serif;}

/* ---------- root palette ---------- */
:root {
    --bg:       #0B0F1A;
    --surface:  #131929;
    --border:   #1F2D45;
    --accent1:  #6C63FF;
    --accent2:  #00D4FF;
    --danger:   #FF4B4B;
    --success:  #00C896;
    --warn:     #FFB347;
    --text:     #E8EDF5;
    --muted:    #8B97B4;
}

/* ---------- page background ---------- */
.stApp {background: var(--bg) !important;}
section[data-testid="stSidebar"] {background: var(--surface) !important; border-right: 1px solid var(--border);}

/* ---------- hide default Streamlit chrome ---------- */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding: 1.5rem 2rem 3rem 2rem;}

/* ---------- sidebar nav buttons ---------- */
div[data-testid="stSidebar"] .stRadio > label {color: var(--muted) !important; font-size: 0.75rem; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.25rem;}
div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {gap: 0.25rem;}
div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    color: var(--text) !important;
    font-size: 0.92rem;
    cursor: pointer;
    transition: all 0.2s;
}
div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {background: rgba(108,99,255,0.12); border-color: rgba(108,99,255,0.4);}

/* ---------- metric tiles ---------- */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(108,99,255,.08), transparent 60%);
}
.metric-label {color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.35rem;}
.metric-value {color: var(--text); font-size: 2rem; font-weight: 700; line-height: 1;}
.metric-delta {font-size: 0.78rem; margin-top: 0.3rem;}
.metric-delta.pos {color: var(--success);}
.metric-delta.neg {color: var(--danger);}
.metric-icon {position: absolute; right: 1.25rem; top: 1rem; font-size: 1.8rem; opacity: .25;}

/* ---------- section headers ---------- */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.25rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
}
.section-header h2 {color: var(--text); font-size: 1.2rem; font-weight: 600; margin: 0;}
.section-header .badge {background: rgba(108,99,255,.18); color: var(--accent1); border-radius: 20px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600;}

/* ---------- model cards ---------- */
.model-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color .25s, transform .2s;
}
.model-card:hover {border-color: var(--accent1); transform: translateY(-2px);}
.model-card.best {border-color: var(--accent1); box-shadow: 0 0 24px rgba(108,99,255,.2);}
.model-name {font-size: 1rem; font-weight: 600; color: var(--text); margin-bottom: 0.75rem;}
.badge-best {background: linear-gradient(90deg, var(--accent1), var(--accent2)); color: #fff; border-radius: 20px; padding: 2px 10px; font-size: 0.72rem; font-weight: 700; margin-left: 0.5rem;}

/* ---------- bar fill (model metrics) ---------- */
.bar-row {display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.45rem;}
.bar-label {color: var(--muted); font-size: 0.78rem; width: 72px; flex-shrink: 0;}
.bar-track {flex: 1; background: rgba(255,255,255,.06); border-radius: 9px; height: 8px; overflow: hidden;}
.bar-fill {height: 100%; border-radius: 9px;}
.bar-pct {color: var(--text); font-size: 0.78rem; font-weight: 600; width: 40px; text-align: right; flex-shrink: 0;}

/* ---------- prediction result ---------- */
.result-fraud {
    background: rgba(255,75,75,.08);
    border: 1px solid rgba(255,75,75,.35);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}
.result-legit {
    background: rgba(0,200,150,.08);
    border: 1px solid rgba(0,200,150,.35);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}
.result-title {font-size: 1.5rem; font-weight: 700; margin-bottom: 0.25rem;}
.result-sub   {color: var(--muted); font-size: 0.88rem;}

/* ---------- risk bar ---------- */
.risk-bar-track {
    background: rgba(255,255,255,.07);
    border-radius: 99px;
    height: 18px;
    overflow: hidden;
    margin: 0.75rem 0;
    position: relative;
}
.risk-bar-fill  {height: 100%; border-radius: 99px; transition: width 0.6s ease;}

/* ---------- form inputs ---------- */
div[data-testid="stForm"] {background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1.5rem;}
.stSelectbox > div > div,
.stNumberInput input,
.stSlider > div > div {background: #1A2235 !important; border-radius: 8px !important; color: var(--text) !important;}
.stFormSubmitButton > button {
    background: linear-gradient(90deg, var(--accent1), var(--accent2)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: opacity .2s !important;
    width: 100% !important;
}
.stFormSubmitButton > button:hover {opacity: 0.88 !important;}

/* ---------- tables  ---------- */
.stDataFrame {border-radius: 12px; overflow: hidden;}

/* ---------- dividers ---------- */
hr {border-color: var(--border) !important;}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

BG  = "#0B0F1A"
SRF = "#131929"
ACCENT  = "#6C63FF"
ACCENT2 = "#00D4FF"
DANGER  = "#FF4B4B"
SUCCESS = "#00C896"
MUTED   = "#8B97B4"
TEXT    = "#E8EDF5"

def set_mpl():
    plt.rcParams.update({
        "figure.facecolor": SRF, "axes.facecolor": SRF,
        "axes.edgecolor":  MUTED, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": TEXT, "grid.color": "#1F2D45",
        "grid.linewidth": 0.5, "axes.grid": True,
    })

@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/transactions.csv")
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_metrics():
    try:
        with open("models/metrics.json") as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_resource
def load_objects():
    try:
        return (
            joblib.load("models/scaler.pkl"),
            joblib.load("models/label_encoder.pkl"),
            joblib.load("models/feature_columns.pkl"),
        )
    except Exception:
        return None, None, None

@st.cache_resource
def load_model(name):
    try:
        return joblib.load(f"models/{name.replace(' ', '_').lower()}.pkl")
    except Exception:
        return None

def metric_card(icon, label, value, delta=None, delta_pos=True):
    delta_html = ""
    if delta is not None:
        cls = "pos" if delta_pos else "neg"
        arrow = "▲" if delta_pos else "▼"
        delta_html = f'<div class="metric-delta {cls}">{arrow} {delta}</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def bar_row(label, val, color):
    pct = int(val * 100)
    st.markdown(f"""
    <div class="bar-row">
        <div class="bar-label">{label}</div>
        <div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color};"></div></div>
        <div class="bar-pct">{val:.3f}</div>
    </div>""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
        <div style="font-size:2.4rem;">🛡️</div>
        <div style="color:#E8EDF5;font-size:1.1rem;font-weight:700;margin-top:0.25rem;">FraudSense AI</div>
        <div style="color:#8B97B4;font-size:0.75rem;">Transaction Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio(
        "Dashboard pages",
        ["📊  Overview & Insights", "⚙️  Model Performance", "🔍  Predict Transaction"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""<div style="color:#8B97B4;font-size:0.75rem;text-align:center;">
        Powered by XGBoost · SMOTE · Scikit-learn<br>
        <span style="color:#6C63FF;">FraudSense v2.0</span>
    </div>""", unsafe_allow_html=True)

# ── Load data / models ─────────────────────────────────────────────────────────
df       = load_data()
metrics  = load_metrics()
scaler, le, features = load_objects()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview & Insights
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("## 📊 Overview & Insights", unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

    if df.empty:
        st.error("Dataset not found. Run `python src/dataset_generator.py` first.")
        st.stop()

    total  = len(df)
    fraud  = int(df["is_fraud"].sum())
    legit  = total - fraud
    f_pct  = fraud / total * 100

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("💳", "Total Transactions", f"{total:,}")
    with c2: metric_card("🚨", "Fraudulent", f"{fraud:,}", f"{f_pct:.2f}% of total", delta_pos=False)
    with c3: metric_card("✅", "Legitimate", f"{legit:,}", f"{100-f_pct:.2f}% of total")
    with c4: metric_card("📈", "Avg Amount ($)", f"{df['amount'].mean():,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ───────────────────────────────────────────────────────────
    col_l, col_r = st.columns([1.4, 1], gap="large")

    with col_l:
        st.markdown("""<div class="section-header">
            <h2>Transaction Type Breakdown</h2>
            <span class="badge">By Fraud Status</span>
        </div>""", unsafe_allow_html=True)

        set_mpl()
        fig, ax = plt.subplots(figsize=(8, 4))
        types  = df["type"].unique()
        x      = np.arange(len(types))
        w      = 0.35
        fraud_counts = [df[(df["type"]==t) & (df["is_fraud"]==1)].shape[0] for t in types]
        legit_counts = [df[(df["type"]==t) & (df["is_fraud"]==0)].shape[0] for t in types]

        ax.bar(x - w/2, legit_counts,  w, color=SUCCESS,  alpha=0.9, label="Legitimate")
        ax.bar(x + w/2, fraud_counts,  w, color=DANGER,   alpha=0.9, label="Fraud")
        ax.set_xticks(x)
        ax.set_xticklabels(types, fontsize=9)
        ax.legend(facecolor=SRF, edgecolor=MUTED, labelcolor=TEXT, fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_r:
        st.markdown("""<div class="section-header">
            <h2>Fraud Ratio</h2>
            <span class="badge">Donut View</span>
        </div>""", unsafe_allow_html=True)

        set_mpl()
        fig2, ax2 = plt.subplots(figsize=(4.5, 4))
        sizes  = [legit, fraud]
        colors = [SUCCESS, DANGER]
        wedges, _ = ax2.pie(sizes, colors=colors, startangle=90,
                            wedgeprops=dict(width=0.55, edgecolor=SRF, linewidth=2))
        ax2.text(0, 0, f"{f_pct:.1f}%\nFraud", ha="center", va="center",
                 color=DANGER, fontsize=14, fontweight="700", linespacing=1.4)
        patches = [mpatches.Patch(color=SUCCESS, label="Legitimate"),
                   mpatches.Patch(color=DANGER,  label="Fraud")]
        ax2.legend(handles=patches, loc="lower center", ncol=2,
                   facecolor=SRF, edgecolor=MUTED, labelcolor=TEXT, fontsize=9, framealpha=0)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # ── Amount distribution ───────────────────────────────────────────────────
    st.markdown("""<div class="section-header" style="margin-top:1.5rem;">
        <h2>Transaction Amount Distribution</h2>
        <span class="badge">Log Scale</span>
    </div>""", unsafe_allow_html=True)

    set_mpl()
    fig3, ax3 = plt.subplots(figsize=(12, 3.5))
    for label, color, mask in [("Legitimate", SUCCESS, df["is_fraud"]==0),
                                  ("Fraud",      DANGER,  df["is_fraud"]==1)]:
        vals = df.loc[mask, "amount"].clip(lower=1)
        ax3.hist(np.log10(vals), bins=60, alpha=0.7, color=color, label=label, density=True)
    ax3.set_xlabel("log₁₀(Amount)", fontsize=9)
    ax3.set_ylabel("Density", fontsize=9)
    ax3.legend(facecolor=SRF, edgecolor=MUTED, labelcolor=TEXT, fontsize=9)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # ── Sample data ───────────────────────────────────────────────────────────
    st.markdown("""<div class="section-header" style="margin-top:1.5rem;">
        <h2>Recent Transactions</h2>
        <span class="badge">Latest 50</span>
    </div>""", unsafe_allow_html=True)

    sample = df.head(50).copy()
    sample["is_fraud"] = sample["is_fraud"].map({0: "✅ Legit", 1: "🚨 Fraud"})
    sample["amount"]   = sample["amount"].map(lambda x: f"${x:,.2f}")
    st.dataframe(sample, use_container_width=True, height=320)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown("## ⚙️ Model Performance", unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

    if not metrics:
        st.error("Model metrics not found. Run `python src/model_trainer.py` first.")
        st.stop()

    best_model = metrics.get("Best_Model", "")
    st.markdown(f"""
    <div style="background:rgba(108,99,255,.12);border:1px solid rgba(108,99,255,.35);
                border-radius:12px;padding:1rem 1.5rem;margin-bottom:1.5rem;
                display:flex;align-items:center;gap:.75rem;">
        <span style="font-size:1.6rem;">🏆</span>
        <div>
            <div style="color:#E8EDF5;font-weight:700;font-size:1rem;">
                Best Model: <span style="color:#6C63FF;">{best_model}</span>
            </div>
            <div style="color:#8B97B4;font-size:0.82rem;">
                Selected by highest F1 Score — balanced Precision × Recall
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Comparison chart ─────────────────────────────────────────────────────
    model_names = [k for k in metrics if k != "Best_Model"]
    metric_keys = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    colors_bar  = [ACCENT, ACCENT2, SUCCESS, DANGER, "#FFB347"]

    set_mpl()
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(14, 4))
    for ax, mk, col in zip(axes, metric_keys, colors_bar):
        vals = [metrics[m][mk] for m in model_names]
        bars = ax.barh(model_names, vals, color=col, alpha=.85, height=0.5)
        ax.set_xlim(0, 1.05)
        ax.set_title(mk, fontsize=9, fontweight="600", color=TEXT)
        ax.tick_params(labelsize=8)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=8, color=TEXT)
    fig.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Individual model cards ────────────────────────────────────────────────
    st.markdown("""<div class="section-header">
        <h2>Detailed Scores</h2>
    </div>""", unsafe_allow_html=True)

    bar_colors = {"Accuracy": ACCENT, "Precision": ACCENT2, "Recall": SUCCESS,
                  "F1 Score": DANGER, "ROC AUC": "#FFB347"}

    cols_grid = st.columns(len(model_names))
    for col, name in zip(cols_grid, model_names):
        with col:
            badge = '<span class="badge-best">★ BEST</span>' if name == best_model else ""
            cls   = "model-card best" if name == best_model else "model-card"
            st.markdown(f'<div class="{cls}"><div class="model-name">{name}{badge}</div>', unsafe_allow_html=True)
            for mk in metric_keys:
                bar_row(mk, metrics[name][mk], bar_colors[mk])
            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Predict Transaction
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("## 🔍 Predict Transaction Risk", unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

    if not metrics or scaler is None:
        st.error("Models not found. Run the training pipeline first.")
        st.stop()

    info_col, form_col = st.columns([1, 2], gap="large")

    with info_col:
        st.markdown("""
        <div style="background:#131929;border:1px solid #1F2D45;border-radius:14px;padding:1.5rem;">
            <div style="font-size:2rem;margin-bottom:.5rem;">🔬</div>
            <div style="color:#E8EDF5;font-weight:600;font-size:1rem;margin-bottom:.5rem;">How it works</div>
            <div style="color:#8B97B4;font-size:0.84rem;line-height:1.7;">
                Enter the transaction details on the right. Our <strong style="color:#6C63FF;">XGBoost / RandomForest</strong>
                model — trained on 50,000+ transactions with SMOTE oversampling — will score the risk
                and return a <em>prediction</em> and a <em>probability score</em>.
            </div>
            <hr style="border-color:#1F2D45;margin:1rem 0;">
            <div style="color:#8B97B4;font-size:0.82rem;">
                <div style="margin-bottom:.4rem;">📌 <b style="color:#E8EDF5;">High-risk signals:</b></div>
                <ul style="padding-left:1.2rem;margin:0;line-height:1.9;">
                    <li>Large amount with zero sender balance</li>
                    <li>CASH_OUT or TRANSFER at unusual hours</li>
                    <li>Recipient balance stays exactly zero</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with form_col:
        with st.form("predict_form"):
            st.markdown("#### Transaction Details")
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                tx_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT", "CASH_IN"])
                amount  = st.number_input("Transaction Amount ($)", min_value=0.0, value=15000.0, step=500.0, format="%.2f")
                oldbalanceOrg = st.number_input("Sender — Old Balance ($)", min_value=0.0, value=15000.0, step=100.0, format="%.2f")
            with r1c2:
                newbalanceOrig = st.number_input("Sender — New Balance ($)", min_value=0.0, value=0.0, step=100.0, format="%.2f")
                oldbalanceDest = st.number_input("Recipient — Old Balance ($)", min_value=0.0, value=0.0, step=100.0, format="%.2f")
                newbalanceDest = st.number_input("Recipient — New Balance ($)", min_value=0.0, value=15000.0, step=100.0, format="%.2f")

            time_of_day = st.slider("Time of Day (24-hour)", 0, 23, 2,
                                    help="Transactions at night (0–4 AM) are higher risk.")

            submitted = st.form_submit_button("🔍  Analyze Transaction Risk", use_container_width=True)

        if submitted:
            with st.spinner("Running fraud analysis…"):
                # Encode & scale
                inp = pd.DataFrame([{
                    "type": tx_type, "amount": amount,
                    "oldbalanceOrg": oldbalanceOrg, "newbalanceOrig": newbalanceOrig,
                    "oldbalanceDest": oldbalanceDest, "newbalanceDest": newbalanceDest,
                    "time_of_day": time_of_day,
                }])
                inp["type"] = le.transform([tx_type])[0] if tx_type in le.classes_ else 0
                inp = inp[features]
                scaled = scaler.transform(inp)

                best_name = metrics.get("Best_Model", "XGBoost")
                model     = load_model(best_name)

                if model is None:
                    st.error("Could not load model file.")
                else:
                    pred  = model.predict(scaled)[0]
                    prob  = model.predict_proba(scaled)[0][1] if hasattr(model, "predict_proba") else float(pred)
                    pct   = prob * 100

                    # ── Risk colour gradient ──────────────────────────────────
                    if pct < 30:
                        bar_col = SUCCESS
                    elif pct < 65:
                        bar_col = "#FFB347"
                    else:
                        bar_col = DANGER

                    st.markdown("<br>", unsafe_allow_html=True)
                    if pred == 1:
                        st.markdown(f"""
                        <div class="result-fraud">
                            <div class="result-title" style="color:#FF4B4B;">🚨 FRAUDULENT TRANSACTION</div>
                            <div class="result-sub">This transaction is flagged as high risk. Review immediately.</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-legit">
                            <div class="result-title" style="color:#00C896;">✅ LEGITIMATE TRANSACTION</div>
                            <div class="result-sub">Transaction appears safe based on current risk model.</div>
                        </div>""", unsafe_allow_html=True)

                    # Risk bar
                    st.markdown(f"""
                    <div style="margin-top:1.25rem;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
                            <span style="color:#8B97B4;font-size:0.82rem;">Risk Score</span>
                            <span style="color:{bar_col};font-weight:700;font-size:0.95rem;">{pct:.1f}%</span>
                        </div>
                        <div class="risk-bar-track">
                            <div class="risk-bar-fill" style="width:{pct:.1f}%;background:linear-gradient(90deg,{bar_col}88,{bar_col});"></div>
                        </div>
                        <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#8B97B4;">
                            <span>Low Risk</span><span>High Risk</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Detail chips
                    st.markdown("<br>", unsafe_allow_html=True)
                    d1, d2, d3 = st.columns(3)
                    with d1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Model Used</div>
                            <div style="color:#6C63FF;font-weight:600;font-size:0.95rem;">{best_name}</div>
                        </div>""", unsafe_allow_html=True)
                    with d2:
                        risk_label = "Low" if pct < 30 else ("Medium" if pct < 65 else "High")
                        col_label  = SUCCESS if pct < 30 else ("#FFB347" if pct < 65 else DANGER)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Risk Level</div>
                            <div style="color:{col_label};font-weight:600;font-size:0.95rem;">{risk_label}</div>
                        </div>""", unsafe_allow_html=True)
                    with d3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Confidence</div>
                            <div style="color:#E8EDF5;font-weight:600;font-size:0.95rem;">{max(pct, 100-pct):.1f}%</div>
                        </div>""", unsafe_allow_html=True)
