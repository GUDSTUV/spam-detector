# ============================================================
# Email Spam Classifier — Public App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

st.set_page_config(
    page_title="Spam Detector AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; letter-spacing: -0.01em; }
.stApp { background: #000000; color: #ededed; }
[data-testid="stSidebar"] { background: #0a0a0a !important; border-right: 1px solid #222222; }
.hero {
    background: #0a0a0a;
    border: 1px solid #222222; border-radius: 12px;
    padding: 2.5rem 3rem; margin-bottom: 2rem;
}
.hero-title { font-size: 2.2rem; font-weight: 700; color: #ffffff; margin: 0 0 0.4rem 0; letter-spacing: -0.03em; }
.hero-sub { font-size: 1rem; color: #888888; margin: 0; font-weight: 400; }
.badge-spam {
    display: inline-block;
    background: #2a0000;
    color: #ff4444; border: 1px solid #ff0000;
    border-radius: 6px; padding: 0.4rem 1rem;
    font-size: 0.9rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;
}
.badge-ham {
    display: inline-block;
    background: #002211;
    color: #00ff88; border: 1px solid #00cc66;
    border-radius: 6px; padding: 0.4rem 1rem;
    font-size: 0.9rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;
}
.conf-bar-wrap { background: #222222; border-radius: 6px; height: 8px; margin: 0.6rem 0; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 6px; transition: width 0.6s ease; }
.metric-tile { background: #0a0a0a; border: 1px solid #222222; border-radius: 8px; padding: 1.2rem 1.5rem; text-align: center; transition: transform 0.2s; }
.metric-tile:hover { transform: translateY(-2px); border-color: #444444; }
.metric-value { font-size: 2rem; font-weight: 700; color: #ffffff; line-height: 1; }
.metric-label { font-size: 0.75rem; color: #888888; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.4rem; }
.section-title {
    font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase; color: #888888;
    margin-bottom: 1.2rem; font-weight: 600;
}
.stTextArea textarea {
    background: #0a0a0a !important; border: 1px solid #222222 !important;
    border-radius: 8px !important; color: #ededed !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.95rem !important;
    transition: border-color 0.2s;
}
.stTextArea textarea::placeholder { color: #555555 !important; opacity: 1 !important; }
.stTextArea textarea:focus { border-color: #ffffff !important; box-shadow: none !important; }
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
.stTabs [data-baseweb="tab-list"] { background: #0a0a0a; border-radius: 8px; gap: 20px; padding: 8px; border: 1px solid #222222; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #888888 !important;
    border-radius: 6px !important; font-family: 'Inter', sans-serif !important; font-weight: 500 !important;
}
.stTabs [aria-selected="true"] { background: #222222 !important; color: #ffffff !important; padding: 8px 16px; }
.stDataFrame { border-radius: 8px; overflow: hidden; border: 1px solid #222222; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Train model (cached & shared with admin page) ─────────
@st.cache_resource
def train_model():
    if not os.path.exists("spam.csv"):
        st.error("spam.csv not found. Please place the dataset in the same folder as app.py.")
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
    model = MultinomialNB(alpha=0.1)
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"], output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    return model, vec, df, acc, cm, report, fpr, tpr, roc_auc

model, vectorizer, df, accuracy, cm, report, fpr, tpr, roc_auc = train_model()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:1.2rem; font-weight:700; color:#ffffff; letter-spacing:1px;'>SPAM DETECTOR</div>
    </div>
    
    <div class='section-title' style='margin-top:1rem;'>About</div>
    <div style='font-size:0.85rem; color:#888888; line-height:1.6;'>
    This app instantly analyses emails and SMS messages to protect you from <b style='color:#ffffff;'>spam</b> and <b style='color:#ffffff;'>phishing</b> attempts. 
    <br><br>
    Simply paste a message to get a real-time security check!
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-title'>Email Spam Detector</div>
    <div class='hero-sub'>
        AI-powered real-time protection against spam and phishing attempts.
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["  🔍  Detect Spam  ", "  📋  Batch Test  "])

# ──────────────────────────────────────────────────────────
# TAB 1 — DETECT SPAM
# ──────────────────────────────────────────────────────────
with tab1:
    col_input, col_result = st.columns([3, 2], gap="large")

    with col_input:
        st.markdown("<div class='section-title'>Paste Email Content</div>", unsafe_allow_html=True)
        email_input = st.text_area(
            label="",
            value=st.session_state.get("email_input", ""),
            placeholder="Paste or type your email content here...\n\nExample: Congratulations! You have won a FREE iPhone. Click here to claim now!",
            height=220, key="email_input", label_visibility="collapsed"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        analyse_btn = st.button("Analyse Email", use_container_width=True)

        st.markdown("<div class='section-title' style='margin-top:1.2rem;'>Quick Examples</div>", unsafe_allow_html=True)
        examples = {
            "🚨 Typical Spam": "Congratulations! You have won a FREE iPhone. Click here to claim your prize now! Limited time offer.",
            "✉️ Normal Email":  "Hi Sarah, just checking if we are still on for the meeting tomorrow at 3pm. Let me know if the time works.",
            "⚠️ Phishing":      "URGENT: Your account has been compromised. Call 08712460324 immediately to verify your details.",
            "📅 Work Email":    "Can you please send me the updated project files before Friday? Thank you so much.",
        }
        def _set_example(val):
            st.session_state["email_input"] = val
        ex_cols = st.columns(len(examples))
        for c, (label, example_text) in zip(ex_cols, examples.items()):
            c.button(label, key=f"example_{label}", on_click=_set_example, args=(example_text,))

    with col_result:
        st.markdown("<div class='section-title'>Result</div>", unsafe_allow_html=True)
        if analyse_btn and email_input.strip():
            tfidf   = vectorizer.transform([email_input])
            pred    = model.predict(tfidf)[0]
            prob    = model.predict_proba(tfidf)[0]
            conf    = max(prob) * 100
            is_spam = pred == 1
            if is_spam:
                st.markdown("<div class='badge-spam'>⚠ SPAM DETECTED</div>", unsafe_allow_html=True)
                bar_color = "#ff4444"
                verdict   = "This email shows strong indicators of spam. Do not click any links or provide personal information."
            else:
                st.markdown("<div class='badge-ham'>✓ LEGITIMATE EMAIL</div>", unsafe_allow_html=True)
                bar_color = "#00ff88"
                verdict   = "This email appears to be legitimate. The classifier found no significant spam indicators."
            st.markdown(f"""
            <div style='margin-top:1.2rem;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                    <span style='font-size:0.8rem; color:#888888;'>Confidence</span>
                    <span style='font-size:0.85rem; color:#ffffff; font-weight:700;'>{conf:.1f}%</span>
                </div>
                <div class='conf-bar-wrap'>
                    <div class='conf-bar-fill' style='width:{conf}%; background:{bar_color};'></div>
                </div>
            </div>
            <div style='background:#0a0a0a; border:1px solid #222222; border-radius:8px;
                        padding:1rem; margin-top:1rem; font-size:0.85rem; color:#aaaaaa; line-height:1.6;'>
                {verdict}
            </div>
            <div style='margin-top:1.2rem; display:flex; gap:0.8rem;'>
                <div style='flex:1; background:#000000; border:1px solid #222222;
                            border-radius:8px; padding:0.8rem; text-align:center;'>
                    <div style='font-size:1.1rem; color:#ff4444;'>{prob[1]*100:.1f}%</div>
                    <div style='font-size:0.7rem; color:#888888; margin-top:3px;'>Spam probability</div>
                </div>
                <div style='flex:1; background:#000000; border:1px solid #222222;
                            border-radius:8px; padding:0.8rem; text-align:center;'>
                    <div style='font-size:1.1rem; color:#00ff88;'>{prob[0]*100:.1f}%</div>
                    <div style='font-size:0.7rem; color:#888888; margin-top:3px;'>Ham probability</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif analyse_btn:
            st.warning("Please enter some email text first.")
        else:
            st.markdown("""
            <div style='text-align:center; padding:3rem 1rem; color:#374151;'>
                <div style='font-size:3rem; margin-bottom:0.8rem;'>📨</div>
                <div style='font-size:0.9rem;'>Paste an email and click<br>
                <b style='color:#4b5563;'>Analyse Email</b> to get a result</div>
            </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# TAB 2 — BATCH TEST
# ──────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-title'>Test Multiple Emails At Once</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.85rem; color:#6b7280; margin-bottom:1rem;'>
    Enter one email per line. The classifier will analyse all of them at once.
    </div>
    """, unsafe_allow_html=True)
    default_batch = """Congratulations! You have won a FREE iPhone. Click here to claim now!
Hi Sarah, just checking if we are still on for the meeting tomorrow at 3pm.
URGENT: Your account has been compromised. Call 08712460324 immediately.
Can you please send me the project files before Friday? Thank you.
Win 1000 cash! Text WIN to 80085. Free entry, terms apply.
The quarterly report is due by end of business Friday. Please review."""
    batch_input = st.text_area(label="", value=default_batch, height=200, label_visibility="collapsed")
    if st.button("Analyse All Emails", key="batch_btn", use_container_width=True):
        lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
        if lines:
            tfidf_batch = vectorizer.transform(lines)
            preds = model.predict(tfidf_batch)
            probs = model.predict_proba(tfidf_batch)
            results = []
            for i, (line, pred, prob) in enumerate(zip(lines, preds, probs)):
                results.append({
                    "#": i + 1,
                    "Email": line[:80] + ("..." if len(line) > 80 else ""),
                    "Verdict": "🚨 SPAM" if pred == 1 else "✅ HAM",
                    "Confidence": f"{max(prob)*100:.1f}%",
                })
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            spam_count = sum(preds)
            ham_count  = len(preds) - spam_count
            st.markdown(f"""
            <div style='display:flex; gap:1rem; margin-top:1rem;'>
                <div class='metric-tile' style='flex:1;'>
                    <div class='metric-value' style='color:#ff4444;'>{spam_count}</div>
                    <div class='metric-label'>Spam Detected</div>
                </div>
                <div class='metric-tile' style='flex:1;'>
                    <div class='metric-value' style='color:#00ff88;'>{ham_count}</div>
                    <div class='metric-label'>Legitimate</div>
                </div>
                <div class='metric-tile' style='flex:1;'>
                    <div class='metric-value'>{len(preds)}</div>
                    <div class='metric-label'>Total Analysed</div>
                </div>
            </div>
            """, unsafe_allow_html=True)