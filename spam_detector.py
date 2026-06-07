# ============================================================
# Email Spam Classification Using Artificial Intelligence
# Algorithm: Multinomial Naive Bayes + TF-IDF
# Author  : [Your Name]
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

# ─────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────

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


def load_dataset():
    if os.path.exists("spam.csv"):
        df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
        df.columns = ["label", "message"]
        print(f"[INFO] Loaded spam.csv — {len(df)} records")
        return df

    print("[INFO] spam.csv not found. Using built-in sample dataset.")
    messages = (
        [("spam", m) for m in SPAM_MESSAGES]
        + [("ham", m) for m in HAM_MESSAGES]
    )
    df = pd.DataFrame(messages, columns=["label", "message"])
    print(f"[INFO] Built-in dataset — {len(df)} records "
          f"({df['label'].value_counts()['spam']} spam, "
          f"{df['label'].value_counts()['ham']} ham)")
    return df


# ─────────────────────────────────────────
# 2. PREPROCESS & SPLIT
# ─────────────────────────────────────────

def prepare_data(df):
    df = df.dropna(subset=["label", "message"]).copy()
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label_num"],
        test_size=0.2, random_state=42, stratify=df["label_num"]
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────
# 3. VECTORIZE & TRAIN
# ─────────────────────────────────────────

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_features=5000,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)
    return model, vectorizer


# ─────────────────────────────────────────
# 4. EVALUATE
# ─────────────────────────────────────────

def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred  = model.predict(X_test_tfidf)
    y_prob  = model.predict_proba(X_test_tfidf)[:, 1]

    accuracy    = accuracy_score(y_test, y_pred)
    report      = classification_report(y_test, y_pred, target_names=["Ham", "Spam"])
    cm          = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)

    print("\n" + "=" * 55)
    print("           MODEL EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Accuracy  : {accuracy * 100:.2f}%")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print("\n  Classification Report:")
    print(report)

    return y_pred, y_prob, cm, fpr, tpr, roc_auc, accuracy, report


# ─────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────

def save_visualisations(df, cm, fpr, tpr, roc_auc, accuracy, report_str):
    os.makedirs("outputs", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # A: Dataset distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["label"].value_counts()
    colors = ["#4CAF50", "#F44336"]
    bars = ax.bar(counts.index, counts.values, color=colors,
                  edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                str(val), ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_title("Dataset Distribution: Ham vs Spam",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Messages")
    ax.set_xlabel("Category")
    plt.tight_layout()
    plt.savefig("outputs/1_dataset_distribution.png", dpi=150)
    plt.close()

    # B: Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"],
                yticklabels=["Ham", "Spam"],
                linewidths=0.5, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("outputs/2_confusion_matrix.png", dpi=150)
    plt.close()

    # C: ROC Curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color="#1976D2", lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1,
            linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#1976D2")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Naive Bayes Classifier",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("outputs/3_roc_curve.png", dpi=150)
    plt.close()

    # D: Precision / Recall / F1 per class
    lines = [l for l in report_str.split("\n")
             if "Ham" in l or "Spam" in l]
    metric_data = {}
    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            metric_data[parts[0]] = {
                "Precision": float(parts[1]),
                "Recall":    float(parts[2]),
                "F1-Score":  float(parts[3]),
            }

    if metric_data:
        metrics_df = pd.DataFrame(metric_data).T
        x = np.arange(len(metrics_df))
        width = 0.25
        fig, ax = plt.subplots(figsize=(7, 4))
        palette = ["#42A5F5", "#66BB6A", "#FFA726"]
        for i, col in enumerate(["Precision", "Recall", "F1-Score"]):
            ax.bar(x + i * width, metrics_df[col], width,
                   label=col, color=palette[i], edgecolor="white")
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics_df.index, fontsize=11)
        ax.set_ylim([0, 1.15])
        ax.set_title("Precision, Recall & F1-Score by Class",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("Score")
        ax.legend()
        plt.tight_layout()
        plt.savefig("outputs/4_metrics_summary.png", dpi=150)
        plt.close()

    print("\n[INFO] Charts saved to outputs/")


# ─────────────────────────────────────────
# 6. LIVE PREDICTION
# ─────────────────────────────────────────

def predict_email(text, model, vectorizer):
    tfidf      = vectorizer.transform([text])
    prediction = model.predict(tfidf)[0]
    probability = model.predict_proba(tfidf)[0]
    label      = "SPAM" if prediction == 1 else "HAM (Not Spam)"
    confidence = max(probability) * 100
    return label, confidence


# ─────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  Email Spam Classification — Naive Bayes + TF-IDF")
    print("=" * 55)

    df = load_dataset()
    print(f"\n[INFO] Class balance:\n{df['label'].value_counts().to_string()}")

    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"\n[INFO] Training samples : {len(X_train)}")
    print(f"[INFO] Testing samples  : {len(X_test)}")

    print("\n[INFO] Training Multinomial Naive Bayes model ...")
    model, vectorizer = train_model(X_train, y_train)

    y_pred, y_prob, cm, fpr, tpr, roc_auc, accuracy, report_str = \
        evaluate_model(model, vectorizer, X_test, y_test)

    save_visualisations(df, cm, fpr, tpr, roc_auc, accuracy, report_str)

    # Live demo
    print("\n" + "=" * 55)
    print("           LIVE PREDICTION DEMO")
    print("=" * 55)
    test_emails = [
        "Congratulations! You have won a FREE iPhone. Click here to claim your prize now!",
        "Hi Sarah, just checking if we are still on for the meeting tomorrow at 3pm.",
        "URGENT: Your account has been compromised. Call 08712460324 immediately.",
        "Can you please send me the project files before Friday? Thank you.",
        "Win 1000 cash! Text WIN to 80085. Free entry, terms apply.",
    ]
    for email in test_emails:
        label, conf = predict_email(email, model, vectorizer)
        short = (email[:60] + "...") if len(email) > 60 else email
        print(f"\n  Email   : {short}")
        print(f"  Result  : {label}  (Confidence: {conf:.1f}%)")

    print("\n[INFO] Project complete. All outputs saved to outputs/")


if __name__ == "__main__":
    main()
