# 🛡️ Email Spam Detector

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An AI-powered, real-time protection web application designed to instantly analyze emails and SMS messages to detect spam and phishing attempts. Built with **Streamlit** and **Scikit-learn**, it features a modern, clean, and mobile-friendly minimalist interface.

## ✨ Features

- **Real-Time Detection**: Instantly classify messages as legitimate (Ham) or Spam using a trained Machine Learning model.
- **Batch Processing**: Analyze multiple emails simultaneously and view a comprehensive breakdown of the results.
- **Secure Admin Dashboard**: A password-protected `/admin` route providing deep insights into model performance metrics (Accuracy, ROC-AUC, Precision, Recall) with dynamic data visualizations.
- **Mobile-Optimized UI**: Clean flat-design aesthetics with native popups that eliminate layout shifting on smaller screens.
- **Automated Caching**: Model training is cached securely in-memory to ensure lighting-fast inference upon user interaction.

## 🧠 Model Architecture

The core of the application relies on Natural Language Processing (NLP) to classify text:
- **Feature Extraction**: `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) transforms the raw text data into numerical vectors.
- **Classifier**: A `MultinomialNB` (Multinomial Naive Bayes) algorithm is trained on the SMS Spam Collection dataset to accurately distinguish between safe and malicious content.

## 📂 Project Structure

```text
Spamdetector/
│
├── app.py                 # Main Streamlit web application (Public facing)
├── pages/
│   └── admin.py           # Secure admin dashboard for model analytics
│
├── convert.py             # Utility script: SMSSpamCollection → spam.csv
├── SMSSpamCollection      # Raw tab-separated dataset
├── spam.csv               # Processed dataset used for model training
├── logo.png               # Brand logo
└── requirements.txt       # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment.

### Installation

1. **Clone or download** this repository.
2. **Create and activate a virtual environment**:
   
   *Windows (PowerShell):*
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   
   *macOS / Linux:*
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. *(Optional)* **Prepare the dataset**: 
   If `spam.csv` is missing, you can generate it from the raw collection using:
   ```bash
   python convert.py
   ```

### Running the Application

Launch the Streamlit server locally:

```bash
streamlit run app.py
```

- **Public App**: The main application will be available at `http://localhost:8501`.
- **Admin Dashboard**: Navigate directly to `http://localhost:8501/admin` to view model analytics (Default password: `admin123`).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a pull request.

## 📝 License

This project is open-source and available under the MIT License.
