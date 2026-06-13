# ============================================================
# Admin Analytics Page — /admin
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)

st.set_page_config(
    page_title="Admin — Spam Detector",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

ADMIN_PASSWORD = "admin123"  # Change this to your own password

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; letter-spacing: -0.01em; }
.stApp { background: #000000; color: #ededed; }
[data-testid="stSidebar"] { background: #0a0a0a !important; border-right: 1px solid #222222; }
.metric-tile { background: #0a0a0a; border: 1px solid #222222; border-radius: 8px; padding: 1.2rem 1.5rem; text-align: center; transition: transform 0.2s; }
.metric-tile:hover { transform: translateY(-2px); border-color: #444444; }
.metric-value { font-size: 2rem; font-weight: 700; color: #ffffff; line-height: 1; }
.metric-label { font-size: 0.75rem; color: #888888; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.4rem; }
.section-title {
    font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase; color: #888888;
    margin-bottom: 1.2rem; font-weight: 600;
}
.stButton > button {
    background: #ffffff !important;
    color: #000000 !important; border: none !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    font-size: 0.95rem !important; padding: 0.6rem 1rem !important;
    transition: all 0.2s ease !important; width: 100% !important;
}
.stButton > button:hover {
    background: #cccccc !important;
    transform: scale(1.02) !important;
}
.nav-link {
    display: block; padding: 0.8rem 1rem; margin-bottom: 0.5rem;
    border-radius: 8px; color: #aaaaaa; text-decoration: none;
    font-weight: 500; font-size: 0.95rem; background: #0a0a0a;
    border: 1px solid #222222; transition: all 0.2s;
}
.nav-link:hover { background: #111111; color: #ffffff; border-color: #444444; }
.nav-link.active { background: #ffffff; color: #000000; border-color: #ffffff; font-weight: 600; }
.stDataFrame { border-radius: 8px; overflow: hidden; border: 1px solid #222222; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────
if "admin_logged_in" not in st.session_state:
    st.session_state["admin_logged_in"] = False

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.8rem; margin-bottom:0.3rem;'>🔐</div>
        <div style='font-size:1.2rem; font-weight:700; color:#ffffff; letter-spacing:1px;'>ADMIN PANEL</div>
    </div>
    
    <div class='section-title' style='margin-top:1rem;'>Model Specs</div>
    <div style='font-size:0.85rem; color:#888888; line-height:1.6;'>
    This classifier uses <b style='color:#ffffff;'>Multinomial Naive Bayes</b>
    with <b style='color:#ffffff;'>TF-IDF</b> vectorization to detect spam emails in real time.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state["admin_logged_in"]:
        st.markdown("""
        <div style='background:#064e3b; border:1px solid #10b981; border-radius:8px;
                    padding:0.6rem 0.8rem; font-size:0.8rem; color:#6ee7b7; margin-bottom:1rem;'>
            ✓ Logged in as Admin
        </div>
        """, unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
            st.session_state["admin_logged_in"] = False
            st.rerun()

# ── Login gate ────────────────────────────────────────────
if not st.session_state["admin_logged_in"]:
    st.markdown("""
    <div style='max-width:400px; margin: 4rem auto; text-align:center;'>
        <div style='font-size:3.5rem; margin-bottom:1rem;'>🔐</div>
        <div style='font-size:1.8rem; font-weight:700; color:#ffffff; margin-bottom:0.5rem;'>
            Admin Access
        </div>
        <div style='font-size:0.9rem; color:#6b7280; margin-bottom:2rem;'>
            This page is restricted to administrators only.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        pwd = st.text_input("", type="password",
                            placeholder="Enter admin password",
                            label_visibility="collapsed")
        if st.button("Login", use_container_width=True):
            if pwd == ADMIN_PASSWORD:
                st.session_state["admin_logged_in"] = True
                st.rerun()
            else:
                st.markdown("""
                <div style='background:#7f1d1d; border:1px solid #ef4444; border-radius:8px;
                            padding:0.6rem 0.8rem; font-size:0.85rem; color:#fca5a5;
                            text-align:center; margin-top:0.5rem;'>
                    ✗ Incorrect password. Please try again.
                </div>
                """, unsafe_allow_html=True)
    st.stop()

# ── From here: admin only ─────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("spam.csv"):
        st.error("spam.csv not found.")
        st.stop()
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label_num"], test_size=0.2, random_state=42, stratify=df["label_num"]
    )
    vec = TfidfVectorizer(stop_words="english", lowercase=True, max_features=5000)
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)
    mdl  = MultinomialNB(alpha=0.1)
    mdl.fit(X_tr, y_train)
    y_pred = mdl.predict(X_te)
    y_prob = mdl.predict_proba(X_te)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"], output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    return mdl, vec, df, acc, cm, report, fpr, tpr, roc_auc

_, _, df, accuracy, cm, report, fpr, tpr, roc_auc = load_model()

# ── Page header ───────────────────────────────────────────
st.markdown("""
<div style='background: #0a0a0a;
            border: 1px solid #222222; border-radius: 12px;
            padding: 2rem 3rem; margin-bottom: 2rem;'>
    <div style='font-size:2rem; font-weight:700; color:#ffffff; margin-bottom:0.3rem;'>
        📊 Model Analytics Dashboard
    </div>
    <div style='font-size:0.95rem; color:#888888;'>
        Classifier performance metrics · UCI SMS Spam Collection · Naive Bayes + TF-IDF
    </div>
</div>
""", unsafe_allow_html=True)

# ── Metric tiles ──────────────────────────────────────────
st.markdown("<div class='section-title'>Performance Metrics</div>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""<div class='metric-tile'>
        <div class='metric-value'>{accuracy*100:.1f}%</div>
        <div class='metric-label'>Accuracy</div></div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class='metric-tile'>
        <div class='metric-value'>{roc_auc:.3f}</div>
        <div class='metric-label'>ROC-AUC</div></div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class='metric-tile'>
        <div class='metric-value'>{report['Spam']['precision']:.2f}</div>
        <div class='metric-label'>Spam Precision</div></div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""<div class='metric-tile'>
        <div class='metric-value'>{report['Spam']['recall']:.2f}</div>
        <div class='metric-label'>Spam Recall</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────
dark_bg    = "#000000"
ax_color   = "#222222"
text_color = "#ffffff"

col_cm, col_roc = st.columns(2, gap="large")

with col_cm:
    st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(dark_bg)
    ax.set_facecolor(dark_bg)
    hm = sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"],
                linewidths=0.8, linecolor=ax_color, ax=ax,
                cbar_kws={"shrink": 0.65, "pad": 0.02})
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(text_color); lbl.set_fontsize(10)
    try:
        cbar = hm.collections[0].colorbar
        for t in cbar.ax.get_yticklabels():
            t.set_color(text_color)
        cbar.outline.set_edgecolor(ax_color)
    except Exception:
        pass
    cmap_blues = plt.get_cmap("Blues")
    norm = mpl.colors.Normalize(vmin=np.min(cm), vmax=np.max(cm))
    for (i, j), val in np.ndenumerate(cm):
        r, g, b, _ = cmap_blues(norm(val))
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        ax.text(j + 0.5, i + 0.5, str(val), ha="center", va="center",
                color="white" if lum < 0.5 else "#0a0e1a",
                fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", color=text_color, fontsize=10)
    ax.set_ylabel("Actual",    color=text_color, fontsize=10)
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(ax_color); spine.set_linewidth(1.2)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_roc:
    st.markdown("<div class='section-title'>ROC Curve</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(dark_bg)
    ax.set_facecolor(dark_bg)
    ax.plot(fpr, tpr, color="#3b82f6", lw=2.5, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#374151", lw=1, linestyle="--")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#3b82f6")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", color=text_color, fontsize=10)
    ax.set_ylabel("True Positive Rate",  color=text_color, fontsize=10)
    ax.tick_params(colors=text_color)
    ax.legend(facecolor="#111827", edgecolor="#1e2a45", labelcolor=text_color, fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(ax_color)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Dataset Distribution</div>", unsafe_allow_html=True)

col_bar, col_table = st.columns([2, 3], gap="large")

with col_bar:
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor(dark_bg)
    ax.set_facecolor(dark_bg)
    bars = ax.bar(counts.index, counts.values,
                  color=["#10b981", "#ef4444"],
                  edgecolor=dark_bg, linewidth=1.5, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, str(val),
                ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=text_color)
    ax.tick_params(colors=text_color)
    ax.set_ylabel("Count", color=text_color, fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(ax_color)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_table:
    st.markdown("<br>", unsafe_allow_html=True)
    report_df = pd.DataFrame({
        "Class":     ["Ham (Legitimate)", "Spam"],
        "Precision": [f"{report['Ham']['precision']:.3f}", f"{report['Spam']['precision']:.3f}"],
        "Recall":    [f"{report['Ham']['recall']:.3f}",    f"{report['Spam']['recall']:.3f}"],
        "F1-Score":  [f"{report['Ham']['f1-score']:.3f}",  f"{report['Spam']['f1-score']:.3f}"],
        "Support":   [int(report['Ham']['support']),        int(report['Spam']['support'])],
    })
    st.dataframe(report_df, use_container_width=True, hide_index=True)