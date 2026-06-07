# ============================================================
# Email Spam Classifier — Streamlit Web App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# Prefer Poppins for matplotlib (falls back to system fonts if unavailable)
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Poppins', 'DejaVu Sans', 'Arial']
})
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1424 !important;
    border-right: 1px solid #1e2540;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0f1424 0%, #141b35 50%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Poppins', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1rem;
    color: #8892aa;
    margin: 0;
    font-weight: 300;
}

/* Cards */
.card {
    background: #111827;
    border: 1px solid #1e2a45;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}

/* Result badges */
.badge-spam {
    display: inline-block;
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    color: #fca5a5;
    border: 1px solid #ef4444;
    border-radius: 50px;
    padding: 0.5rem 1.5rem;
    font-family: 'Poppins', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 2px;
}
.badge-ham {
    display: inline-block;
    background: linear-gradient(135deg, #064e3b, #065f46);
    color: #6ee7b7;
    border: 1px solid #10b981;
    border-radius: 50px;
    padding: 0.5rem 1.5rem;
    font-family: 'Poppins', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 2px;
}

/* Confidence bar */
.conf-bar-wrap {
    background: #1e2540;
    border-radius: 50px;
    height: 10px;
    margin: 0.6rem 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 0.6s ease;
}

/* Metric tiles */
.metric-tile {
    background: #111827;
    border: 1px solid #1e2a45;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value {
    font-family: 'Poppins', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #60a5fa;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.4rem;
}

/* Section headers */
.section-title {
    font-family: 'Poppins', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #3b82f6;
    border-bottom: 1px solid #1e2a45;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}

/* Input area override */
.stTextArea textarea {
    background: #111827 !important;
    border: 1px solid #1e2a45 !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    caret-color: #e8eaf0 !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea::placeholder {
    color: #9ca3af !important;
    opacity: 1 !important;
}
.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.4) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    gap: 20px;
    padding: 10px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6b7280 !important;
    border-radius: 8px !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: #1e2a45 !important;
    color: #60a5fa !important;
    padding: 10px;
}

/* Table */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Hide Streamlit branding but keep header so sidebar toggle remains visible */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Dataset ───────────────────────────────────────────────
SPAM_MESSAGES = [
    "Congratulations! You won a FREE iPhone. Click here to claim now!",
    "URGENT: Your bank account has been compromised. Call us immediately.",
    "You have been selected for a gift card worth 1000. Claim today!",
    "FREE entry in our weekly competition. Text WIN to 80085 now!",
    "WINNER!! As a valued network customer you have been selected.",
    "SIX chances to win CASH! From 100 to 20000 pounds txt CSH11.",
    "URGENT! Your Mobile number has been awarded a 2000 Bonus Caller Prize.",
    "Had your contract mobile 11 months? Update to the latest Camera Phone FREE.",
    "You are a winner specially selected to receive 1000 cash or prize.",
    "Todays Offer: Get 50 percent OFF on all products. Limited time only!",
    "Dear friend, I am writing to inform you of a business opportunity.",
    "You have won the international lottery. Send details for bank transfer.",
    "Your account will be suspended. Verify now at our secure website link.",
    "Buy cheap meds online! No prescription needed. 70 percent OFF today.",
    "Meet hot singles in your area! Free signup, no credit card required.",
    "Make 5000 a week from home! No experience needed. Start today!",
    "Your parcel is waiting. Pay customs fee to release your package now.",
    "Mobile update! You are entitled to upgrade for a FREE camera phone now.",
    "Call from landline. You have won a 2 night cruise for two people.",
    "Did you hear the good news? You can earn money fast from home easily!",
    "Act now! This offer expires in 24 hours. Do not miss out on this deal!",
    "Get approved for a loan up to ten thousand dollars. No credit check.",
    "Exclusive deal for YOU only. Click the link to save big today!",
    "Free ringtone for your mobile. Reply YES to claim your prize immediately.",
    "Final notice: Collect your prize before the deadline passes tonight.",
    "Win a brand new car! Enter our draw today. No purchase necessary.",
    "Limited spots available! Sign up now for free money paid every week.",
    "You qualify for a government grant. No repayment required. Apply now!",
    "Earn cash every time you refer a friend to our great network today.",
    "Your credit score has improved. Take advantage with a new card offer.",
    "Hello dear, I am a Nigerian prince and need your urgent assistance.",
    "You are pre-approved for a large credit line. Accept today urgently.",
    "Stop wasting money! Switch to our plan and save instantly today.",
    "Breaking news: You are the millionth visitor. Collect your prize!",
    "Your subscription has been renewed. Call us to cancel your account.",
    "New message from a secret admirer waiting! Login to read it right now.",
    "Work from home opportunity. Earn daily. No experience needed!",
    "You are a lucky winner! Kindly call back to claim your reward today.",
    "Double your investment in 7 days! Guaranteed returns on every deposit.",
    "Last chance: Redeem your free prize before midnight tonight.",
    "Trouble with debt? We can help you write it all off legally and fast.",
    "This is your final notice regarding your vehicle warranty expiration.",
    "Click here to update your payment information before account is locked.",
    "You have been randomly selected for our exclusive customer prize draw.",
    "Hot deals available now! Lowest prices guaranteed or your money back.",
    "Earn unlimited income from home. No boss, no schedule, just freedom.",
    "Your prize is ready to be collected. Reply CLAIM to receive it.",
    "Get rich quick with our proven system. Thousands already earning daily.",
    "Your free gift is waiting. Respond now before it is given away.",
    "Special promotion: Buy one get one free on all our amazing products.",
]

HAM_MESSAGES = [
    "Hey, are you free to meet for lunch tomorrow?",
    "Do not forget the team meeting is at 3pm today.",
    "I will pick up the kids from school. See you at 6.",
    "Can you send me the report you mentioned yesterday?",
    "The project deadline has been moved to next Friday.",
    "Happy birthday! Hope you have a wonderful day.",
    "Thanks for helping me out last week, I really appreciate it.",
    "Are you watching the game tonight?",
    "I am running about 10 minutes late, sorry!",
    "Can we reschedule our call to Thursday instead?",
    "Just checking in. How did the interview go?",
    "I finished the assignment. Let me know if you need a copy.",
    "Dinner was amazing last night. Thanks for organizing it!",
    "The new policy documents are attached. Please review them.",
    "Good morning! Hope you have a productive day.",
    "Can you forward me the agenda for tomorrows meeting?",
    "I got your voicemail. I will call you back after 5pm.",
    "Just left the office. Traffic is terrible right now.",
    "The client approved the proposal. Great news for the team!",
    "Please confirm your attendance for the annual conference.",
    "I need to reschedule our appointment for next week.",
    "The weather is terrible today. Stay safe on the roads.",
    "Reminder: your dentist appointment is tomorrow at 10am.",
    "Can you help me proofread this email before I send it?",
    "I found a great recipe for pasta. Want me to send it?",
    "The library books are due back by Friday afternoon.",
    "Let me know when you arrive and I will come down to meet you.",
    "Great work on the presentation today. The client loved it.",
    "I will be working from home tomorrow if you need to reach me.",
    "Can you pick up some milk on the way home tonight?",
    "The Wi-Fi password is on the back of the router.",
    "I am heading to the gym after work. Join me if free.",
    "Your package has been delivered to the front desk today.",
    "The quarterly report is due by end of business Friday.",
    "Hi, just wanted to check if you received my previous email.",
    "The meeting has been moved to the conference room on floor 3.",
    "Thanks for the feedback. I will revise and send it again.",
    "We need to discuss the budget before approving the plan.",
    "Your flight is confirmed. Check-in opens 24 hours before.",
    "The office will be closed on Monday for the public holiday.",
    "I am at the coffee shop near your office. Want to meet?",
    "Let me know which dates work best for the next sprint review.",
    "Could you review the attached contract before signing it?",
    "I will send you the login details once the account is set up.",
    "The school play is next Thursday at 7pm. Hope to see you.",
    "Please bring your ID to the appointment tomorrow morning.",
    "I have already paid the invoice. Here is the receipt attached.",
    "Looking forward to seeing you at the conference next week.",
    "Hope you are feeling better. Let us know if you need anything.",
    "Your order has been dispatched and will arrive in 2 days.",
    "Can we push the deadline by 48 hours? I need a bit more time.",
    "The annual review forms need to be submitted by December 1st.",
    "Confirming our plans for the weekend. We are still on!",
    "The new employee induction starts at 9am in the main hall.",
    "Please complete the survey by end of the week.",
    "I accidentally sent the wrong file. Please disregard last email.",
    "Your CV looks great! We would like to invite you for an interview.",
    "The system maintenance window is tonight from 11pm to 2am.",
    "I booked the restaurant for 7:30pm. Looking forward to it.",
    "Could you double-check the figures in the budget spreadsheet?",
    "We are planning a small farewell event for Sarah on Friday.",
    "The new office supplies have arrived and are in the storage room.",
    "Please log your hours before the end of the pay period.",
    "Our team won the internal hackathon! So proud of everyone.",
    "The training session has been rescheduled to next Tuesday.",
    "I have shared the Google Doc with you. You should have access.",
    "Can you give me a call when you get a chance? Nothing urgent.",
    "I have submitted the travel request. Awaiting approval now.",
    "The venue for the workshop has been confirmed for next month.",
    "Please review the agenda and add any items you want to discuss.",
    "Let me know if you need extra time to complete the task.",
    "I found your keys on the kitchen counter this morning.",
    "The plumber is coming between 9am and 12pm tomorrow.",
    "Your health insurance renewal is due next month. Please check.",
    "Have a safe trip! See you when you get back next week.",
    "The kids had a great time at the birthday party. Thank you!",
    "I attached the PDF you asked for. Let me know if it works.",
    "The building fire drill is at 11am today. Please take part.",
    "Can you confirm you received the updated project schedule?",
    "We are moving to a new CRM system starting next quarter.",
    "I will be off on annual leave from the 15th to the 22nd.",
    "Please update your emergency contact details in the HR portal.",
    "The code review comments have been addressed. Ready to merge.",
    "Lunch is on me today. Pick a place and I will book it for us.",
    "I saw your presentation. The data visualization was excellent.",
    "Thank you for your patience while we resolved the system issue.",
    "The water has been restored. Everything is back to normal now.",
    "Your test results came back normal. Nothing to worry about.",
    "The IT department will upgrade our systems this weekend.",
    "I am outside. Please buzz me in when you are ready.",
    "The committee approved the funding proposal. Congratulations!",
    "Please provide your bank details for the reimbursement payment.",
    "Can you send me the address for the event on Saturday please?",
    "The draft is ready for your review before we publish it.",
    "Congratulations on your promotion! You really deserve it.",
    "The new student orientation is on Monday at 10am in the hall.",
    "I will need the report by Thursday morning at the latest.",
    "Your feedback was really helpful. I will make those changes.",
    "We are meeting for drinks after work on Friday. Please join us!",
    "The bus route has changed temporarily because of road works.",
    "Your password has been reset. Check your email for the link.",
    "I am attending the conference and will be back on Wednesday.",
    "The printer on the second floor is out of toner again today.",
    "Just finished reading the book you recommended. Loved it!",
    "The deposit has been transferred. Please confirm receipt.",
    "Could you cover my shift on Saturday? I will owe you one.",
    "The new intern starts on Monday. Please make them feel welcome.",
    "I have included all the amendments in the updated version.",
    "The gas meter reading needs to be submitted by the 28th.",
    "Your library card is due for renewal. Visit the library.",
    "We need at least three quotes before approving the purchase.",
    "I am really proud of how the team handled that difficult situation.",
    "The schedule for next semester has been posted on the portal.",
    "Your parking permit expires at the end of this month.",
    "I made dinner reservations for our anniversary. It is a surprise!",
    "The backup generator test will run at 2pm this afternoon.",
    "Let us sync up tomorrow morning before the important client call.",
    "I have read the document and have some comments to discuss.",
    "Thanks again for the recommendation. I actually got the job!",
    "The board meeting minutes have been circulated for approval.",
    "I forgot my badge. Can you let me in through the side door?",
    "We got the green light to proceed with phase two of the project.",
    "Your visa application has been received and is under review.",
    "The supply delivery is scheduled for Thursday afternoon.",
    "I am putting together a reading list for the team. Any suggestions?",
    "The internship program applications close on the 30th of June.",
    "Can you review the tender document before we submit it?",
    "I have attached the meeting notes. Please check the action items.",
    "The new coffee machine is finally working. Life is better now.",
    "We need to discuss the Q3 projections together before Friday.",
    "Your passport has been renewed and is ready for collection.",
    "I have already replied to their email. No need to follow up.",
    "The sports day is next Saturday. Hope you and family can make it.",
    "Can you confirm which document version I should be working from?",
    "The client wants to move the launch date forward by two weeks.",
    "The school trip permission forms are due back by Monday morning.",
    "I have created a shared folder for all the project files online.",
    "We should grab coffee soon and catch up properly with each other.",
    "The app update is live. Let me know if you spot any bugs.",
    "Please activate your new bank card before the end of the week.",
    "I am available for the 2pm slot if that still works for you.",
    "The meeting room booking system has been updated. Check new link.",
    "The report looks good overall but needs a few minor tweaks.",
    "I spoke to the supplier and they can deliver by Thursday.",
    "Can you check if the server is still running? It seemed slow.",
    "Your new laptop will be ready for pickup from IT by tomorrow.",
    "We are renewing the lease on the office space for two more years.",
    "The team building event is confirmed for the 18th next month.",
    "Please let me know which modules you plan to take next semester.",
    "The conference call link has been updated. Check the calendar.",
    "I completed the onboarding checklist. Who do I send it to?",
    "Can someone help me move the boxes from storage to the office?",
]

# ── Train model (cached) ──────────────────────────────────
@st.cache_resource
def train_model():
    if os.path.exists("spam.csv"):
        df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
        df.columns = ["label", "message"]
    else:
        messages = (
            [("spam", m) for m in SPAM_MESSAGES] +
            [("ham",  m) for m in HAM_MESSAGES]
        )
        df = pd.DataFrame(messages, columns=["label", "message"])

    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label_num"],
        test_size=0.2, random_state=42, stratify=df["label_num"]
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
    report = classification_report(y_test, y_pred, target_names=["Ham","Spam"], output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    return model, vec, df, acc, cm, report, fpr, tpr, roc_auc

model, vectorizer, df, accuracy, cm, report, fpr, tpr, roc_auc = train_model()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.8rem; margin-bottom:0.3rem;'></div>
        <div style='font-family: Poppins, sans-serif; font-size:0.9rem;
                color:#60a5fa; letter-spacing:2px;'>SPAM DETECTOR</div>
        <div style='font-size:0.7rem; color:#4b5563; margin-top:0.2rem;'>
            Naive Bayes + TF-IDF
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Model Stats</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='metric-tile' style='margin-bottom:0.8rem;'>
        <div class='metric-value'>{accuracy*100:.1f}%</div>
        <div class='metric-label'>Accuracy</div>
    </div>
    <div class='metric-tile' style='margin-bottom:0.8rem;'>
        <div class='metric-value'>{roc_auc:.3f}</div>
        <div class='metric-label'>ROC-AUC</div>
    </div>
    <div class='metric-tile'>
        <div class='metric-value'>{len(df)}</div>
        <div class='metric-label'>Training Samples</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>About</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.8rem; color:#6b7280; line-height:1.6;'>
    This classifier uses <b style='color:#93c5fd;'>Multinomial Naive Bayes</b>
    with <b style='color:#93c5fd;'>TF-IDF</b> vectorization to detect
    spam emails in real time.
    <br><br>
    Place <code style='background:#1e2540; padding:2px 6px; border-radius:4px;
    color:#93c5fd;'>spam.csv</code> in the same folder to train on the full
    UCI dataset for even higher accuracy.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Appearance</div>", unsafe_allow_html=True)
    theme_choice = st.radio("Theme", ("Black", "White"), index=0, key="theme_choice")

# Apply theme overrides chosen in the sidebar (Black or White)
theme = st.session_state.get("theme_choice", "Black")
if theme == "White":
    bg = "#ffffff"
    text = "#000000"
    card = "#f3f4f6"
    sidebar_bg = "#f8fafc"
    btn_bg = "#000000"
    btn_text = "#ffffff"
else:
    bg = "#0a0e1a"
    text = "#e8eaf0"
    card = "#111827"
    sidebar_bg = "#0f1424"
    btn_bg = "linear-gradient(135deg, #1d4ed8, #2563eb)"
    btn_text = "#ffffff"

# ── Main content ──────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-title'>Email Spam Classifier</div>
    <div class='hero-sub'>
        AI-powered detection using Multinomial Naive Bayes &amp; TF-IDF &nbsp;·&nbsp;
        Python &amp; Scikit-learn
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["  🔍  Detect Spam  ", "  📊  Model Analytics  ", "  📋  Batch Test  "])

# ──────────────────────────────────────────────────────────
# TAB 1 — DETECT
# ──────────────────────────────────────────────────────────
with tab1:
    col_input, col_result = st.columns([3, 2], gap="large")

    with col_input:
        st.markdown("<div class='section-title'>Paste Email Content</div>", unsafe_allow_html=True)

        email_input = st.text_area(
            label="",
            value=st.session_state.get("email_input", ""),
            placeholder="Paste or type your email content here...\n\nExample: Congratulations! You have won a FREE iPhone. Click here to claim now!",
            height=220,
            key="email_input",
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        analyse_btn = st.button("Analyse Email", use_container_width=True)

        st.markdown("<div class='section-title' style='margin-top:1.2rem;'>Quick Examples</div>", unsafe_allow_html=True)
        examples = {
            "🚨 Typical Spam": "Congratulations! You have won a FREE iPhone. Click here to claim your prize now! Limited time offer.",
            "✉️ Normal Email": "Hi Sarah, just checking if we are still on for the meeting tomorrow at 3pm. Let me know if the time works.",
            "⚠️ Phishing": "URGENT: Your account has been compromised. Call 08712460324 immediately to verify your details.",
            "📅 Work Email": "Can you please send me the updated project files before Friday? Thank you so much.",
        }

        def _set_example(val):
            st.session_state["email_input"] = val

        ex_cols = st.columns(len(examples))
        for c, (label, example_text) in zip(ex_cols, examples.items()):
            c.button(label, key=f"example_{label}", on_click=_set_example, args=(example_text,))

    with col_result:
        st.markdown("<div class='section-title'>Result</div>", unsafe_allow_html=True)

        if analyse_btn and email_input.strip():
            tfidf    = vectorizer.transform([email_input])
            pred     = model.predict(tfidf)[0]
            prob     = model.predict_proba(tfidf)[0]
            conf     = max(prob) * 100
            is_spam  = pred == 1

            if is_spam:
                st.markdown("<div class='badge-spam'>⚠ SPAM DETECTED</div>", unsafe_allow_html=True)
                bar_color = "#ef4444"
                verdict   = "This email shows strong indicators of spam. Do not click any links or provide personal information."
            else:
                st.markdown("<div class='badge-ham'>✓ LEGITIMATE EMAIL</div>", unsafe_allow_html=True)
                bar_color = "#10b981"
                verdict   = "This email appears to be legitimate. The classifier found no significant spam indicators."

            st.markdown(f"""
            <div style='margin-top:1.2rem;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                    <span style='font-size:0.8rem; color:#6b7280;'>Confidence</span>
                    <span style='font-family: Poppins, sans-serif; font-size:0.85rem;
                                 color:{text}; font-weight:700;'>{conf:.1f}%</span>
                </div>
                <div class='conf-bar-wrap'>
                    <div class='conf-bar-fill'
                         style='width:{conf}%; background:{bar_color};'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background:#0f1424; border:1px solid #1e2a45; border-radius:10px;
                        padding:1rem; margin-top:1rem; font-size:0.85rem; color:#9ca3af;
                        line-height:1.6;'>
                {verdict}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style='margin-top:1.2rem; display:flex; gap:0.8rem;'>
                <div style='flex:1; background:#111827; border:1px solid #1e2a45;
                            border-radius:10px; padding:0.8rem; text-align:center;'>
                    <div style='font-family:Poppins, sans-serif; font-size:1.1rem;
                                color:#60a5fa;'>{prob[1]*100:.1f}%</div>
                    <div style='font-size:0.7rem; color:#6b7280; margin-top:3px;'>
                        Spam probability</div>
                </div>
                <div style='flex:1; background:#111827; border:1px solid #1e2a45;
                            border-radius:10px; padding:0.8rem; text-align:center;'>
                    <div style='font-family:Poppins, sans-serif; font-size:1.1rem;
                                color:#34d399;'>{prob[0]*100:.1f}%</div>
                    <div style='font-size:0.7rem; color:#6b7280; margin-top:3px;'>
                        Ham probability</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif analyse_btn:
            st.warning("Please enter some email text first.")
        else:
            st.markdown("""
            <div style='text-align:center; padding:3rem 1rem; color:#374151;'>
                <div style='font-size:3rem; margin-bottom:0.8rem;'>📨</div>
                <div style='font-size:0.9rem;'>
                    Paste an email and click<br>
                    <b style='color:#4b5563;'>Analyse Email</b> to get a result
                </div>
            </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# TAB 2 — ANALYTICS
# ──────────────────────────────────────────────────────────
with tab2:
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

    col_cm, col_roc = st.columns(2, gap="large")

    # Use theme colors so plots remain readable in both light and dark themes
    dark_bg = bg
    ax_color = "#374151"
    text_color = text

    with col_cm:
        st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(dark_bg)
        ax.set_facecolor(dark_bg)
        hm = sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                    xticklabels=["Ham","Spam"],
                    yticklabels=["Ham","Spam"],
                    linewidths=0.8, linecolor=ax_color, ax=ax,
                    cbar_kws={"shrink": 0.65, "pad": 0.02})

        # Style tick labels (ensure Poppins and visible color)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(text_color)
            lbl.set_fontfamily('Poppins')
            lbl.set_fontsize(10)

        # Configure colorbar ticks/labels to match theme
        try:
            cbar = hm.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color=text_color)
            for t in cbar.ax.get_yticklabels():
                t.set_color(text_color)
                t.set_fontfamily('Poppins')
            cbar.outline.set_edgecolor(ax_color)
            cbar.ax.patch.set_edgecolor(ax_color)
        except Exception:
            pass

        # Annotate cells with contrasting text colors for readability
        cmap = plt.get_cmap("Blues")
        norm = mpl.colors.Normalize(vmin=np.min(cm), vmax=np.max(cm))
        light_text = "white"
        dark_text = "#000000" if theme == "White" else "#0a0e1a"
        for (i, j), val in np.ndenumerate(cm):
            rgba = cmap(norm(val))  # (r,g,b,a) in 0-1
            r, g, b, _ = rgba
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            ann_color = light_text if luminance < 0.5 else dark_text
            ax.text(j + 0.5, i + 0.5, str(val),
                ha="center", va="center", color=ann_color,
                fontsize=14, fontweight="bold")

        ax.set_xlabel("Predicted", color=text_color, fontsize=10)
        ax.set_ylabel("Actual", color=text_color, fontsize=10)
        ax.tick_params(colors=text_color)

        # Emphasize the axes box so it stands out against the background
        for spine in ax.spines.values():
            spine.set_edgecolor(ax_color)
            spine.set_linewidth(1.2)
        ax.patch.set_edgecolor(ax_color)
        ax.patch.set_linewidth(1.2)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_roc:
        st.markdown("<div class='section-title'>ROC Curve</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(dark_bg)
        ax.set_facecolor(dark_bg)
        ax.plot(fpr, tpr, color="#3b82f6", lw=2.5,
                label=f"AUC = {roc_auc:.4f}")
        ax.plot([0,1],[0,1], color="#374151", lw=1, linestyle="--")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#3b82f6")
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
        ax.set_xlabel("False Positive Rate", color=text_color, fontsize=10)
        ax.set_ylabel("True Positive Rate", color=text_color, fontsize=10)
        ax.tick_params(colors=text_color)
        ax.legend(facecolor=card, edgecolor="#1e2a45",
              labelcolor=text_color, fontsize=10)
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
        colors_bar = ["#10b981", "#ef4444"]
        bars = ax.bar(counts.index, counts.values, color=colors_bar,
                  edgecolor=bg, linewidth=1.5, width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2,
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

# ──────────────────────────────────────────────────────────
# TAB 3 — BATCH TEST
# ──────────────────────────────────────────────────────────
with tab3:
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

    batch_input = st.text_area(
        label="",
        value=default_batch,
        height=200,
        label_visibility="collapsed"
    )

    if st.button("Analyse All Emails", key="batch_btn", use_container_width=True):
        lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
        if lines:
            tfidf_batch = vectorizer.transform(lines)
            preds       = model.predict(tfidf_batch)
            probs       = model.predict_proba(tfidf_batch)

            results = []
            for i, (line, pred, prob) in enumerate(zip(lines, preds, probs)):
                results.append({
                    "#":          i + 1,
                    "Email":      line[:80] + ("..." if len(line) > 80 else ""),
                    "Verdict":    "🚨 SPAM" if pred == 1 else "✅ HAM",
                    "Confidence": f"{max(prob)*100:.1f}%",
                })

            results_df = pd.DataFrame(results)
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            spam_count = sum(preds)
            ham_count  = len(preds) - spam_count
            st.markdown(f"""
            <div style='display:flex; gap:1rem; margin-top:1rem;'>
                <div class='metric-tile' style='flex:1;'>
                    <div class='metric-value' style='color:#ef4444;'>{spam_count}</div>
                    <div class='metric-label'>Spam Detected</div>
                </div>
                <div class='metric-tile' style='flex:1;'>
                    <div class='metric-value' style='color:#10b981;'>{ham_count}</div>
                    <div class='metric-label'>Legitimate</div>
                </div>
                <div class='metric-tile' style='flex:1;'>
                    <div class='metric-value'>{len(preds)}</div>
                    <div class='metric-label'>Total Analysed</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
