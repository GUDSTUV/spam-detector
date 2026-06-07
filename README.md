# Spamdetector

Simple Streamlit app and training scripts for SMS/Email spam detection using TF-IDF + MultinomialNB.

## Files
- `app.py` — Streamlit web app (primary entrypoint).
- `spam_detector.py` — standalone training / evaluation script (optional).
- `convert.py` — convert `SMSSpamCollection` → `spam.csv`.
- `SMSSpamCollection` — raw dataset (tab-separated).
- `spam.csv` — CSV dataset used by the app (if present, app will load it).
- `requirements.txt` — runtime dependencies.

## Quick start
1. (Optional) create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. (Optional) If you only have `SMSSpamCollection`, convert it:

```bash
python convert.py
# produces spam.csv
```

4. Run the Streamlit app

```bash
streamlit run app.py
```

## Reproducible environment (optional)
Consider pinning dependency versions in `requirements.txt` for reproducibility. Example pins:

```
streamlit==1.25.0
pandas==2.1.0
numpy==1.26.0
matplotlib==3.8.0
seaborn==0.12.2
scikit-learn==1.3.2
`````



