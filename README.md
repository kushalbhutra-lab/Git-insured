# ⚖️ Insurance Claim Settlement Bias Analyzer

A comprehensive Streamlit dashboard for detecting and analyzing bias in insurance claim settlement processes.

## Features

| Module | Description |
|--------|-------------|
| 📊 Overview & Descriptive | Cross-tabulations, distributions, descriptive stats |
| 🔍 Diagnostic Bias Analysis | Chi-square tests, heatmaps, team/income/age bias detection |
| 🤖 ML Model Training | KNN, Decision Tree, Random Forest, Gradient Boosted with feature engineering |
| 📈 Model Evaluation | Confusion matrices, ROC curves, Precision/Recall/F1, FP/FN analysis |
| 📋 Findings | Automated bias findings & recommendations |

## Quick Start

### Local (Python)

```bash
# 1. Clone or download
git clone <your-repo-url>
cd insurance_bias_analysis

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New App** → Select your repo → Set `app.py` as the main file
4. Click **Deploy** — live in ~2 minutes!

## Dataset

- Built-in demo dataset: 1,500 synthetic insurance claims
- Upload your own CSV via the sidebar uploader
- Required columns (if uploading): `Age_Group`, `Income_Bracket`, `Settlement_Team`, `Policy_Type`, `Claim_Amount`, `Claim_Duration_Days`, `Premium_Amount`, `Num_Prior_Claims`, `Policy_Tenure_Years`, `Gender`, `Region`, `Settlement_Status`

## Bias Flags Embedded in Demo Data

| Bias Type | Effect |
|-----------|--------|
| Team B | -18% settlement probability |
| Low Income | -20% settlement probability |
| Age 60+ | -12% settlement probability |
| High Claim Amount (>P75) | -10% settlement probability |

## ML Models

- **KNN** — K-Nearest Neighbors (configurable k)
- **Decision Tree** — Configurable max depth
- **Random Forest** — Ensemble with configurable estimators
- **Gradient Boosted** — XGB-style gradient boosting

## Author
Insurance Settlement Bias Analysis Project
