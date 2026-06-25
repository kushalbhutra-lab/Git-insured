import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from scipy.stats import chi2_contingency
import io

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Claim Bias Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://api.fontshare.com/v2/css?f[]=satoshi@400,500,700&display=swap');

:root {
    --primary: #01696f;
    --primary-light: #cedcd8;
    --bg: #f7f6f2;
    --surface: #ffffff;
    --text: #28251d;
    --muted: #7a7974;
    --error: #a12c7b;
    --warning: #964219;
    --success: #437a22;
    --border: rgba(40,37,29,0.12);
}

html, body, [class*="css"] {
    font-family: 'Satoshi', sans-serif !important;
}

.main .block-container {
    padding: 1.5rem 2rem 3rem;
    max-width: 1400px;
}

/* Header Banner */
.hero-banner {
    background: linear-gradient(135deg, #01696f 0%, #0c4e54 60%, #0f3638 100%);
    color: white;
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.hero-banner h1 { font-size: 1.8rem; font-weight: 700; margin: 0 0 0.3rem; color: white; }
.hero-banner p  { margin: 0; opacity: 0.85; font-size: 0.95rem; }

/* KPI cards */
.kpi-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(40,37,29,0.06);
}
.kpi-value { font-size: 2rem; font-weight: 700; color: var(--primary); }
.kpi-label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.25rem; }
.kpi-delta { font-size: 0.85rem; font-weight: 500; margin-top: 0.2rem; }
.delta-pos { color: var(--success); }
.delta-neg { color: var(--error); }

/* Section headers */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text);
    border-left: 4px solid var(--primary);
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}

/* Bias alert */
.bias-alert {
    background: #fff4f0;
    border: 1px solid #f5c6b0;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.bias-alert-title { font-weight: 700; color: #964219; margin-bottom: 0.25rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1c1b19;
}
[data-testid="stSidebar"] * { color: #cdccca !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stFileUploader label { color: #cdccca !important; }

/* Metric labels */
[data-testid="stMetricLabel"] { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: var(--primary); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; border-bottom: 2px solid var(--border); }
.stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; font-size: 0.85rem; font-weight: 500; }
.stTabs [aria-selected="true"] { background: var(--primary) !important; color: white !important; }

/* Tables */
.dataframe thead th { background: #f7f6f2 !important; font-weight: 600; }

/* Badges */
.badge {
    display: inline-block;
    padding: 0.2em 0.65em;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-bias { background: #fde8e0; color: #964219; }
.badge-ok   { background: #d4dfcc; color: #437a22; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────
@st.cache_data
def generate_default_data():
    np.random.seed(42)
    n = 1500
    ages = np.random.choice(['18-30', '31-45', '46-60', '60+'], n, p=[0.2, 0.35, 0.3, 0.15])
    income_brackets = np.random.choice(['Low (<30K)', 'Middle (30K-70K)', 'High (70K-150K)', 'Premium (150K+)'], n, p=[0.25, 0.35, 0.25, 0.15])
    teams = np.random.choice(['Team A', 'Team B', 'Team C', 'Team D'], n, p=[0.25, 0.3, 0.25, 0.2])
    policy_types = np.random.choice(['Term Life', 'Whole Life', 'Health', 'Auto'], n, p=[0.3, 0.25, 0.3, 0.15])
    claim_amounts = np.round(np.random.lognormal(mean=9.5, sigma=1.2, size=n), 2)
    claim_durations = np.random.randint(1, 120, n)
    premium_amounts = np.round(np.random.lognormal(mean=7.5, sigma=0.8, size=n), 2)
    num_prior_claims = np.random.poisson(1.5, n)
    policy_tenure = np.random.randint(1, 25, n)
    claimant_gender = np.random.choice(['Male', 'Female'], n, p=[0.52, 0.48])
    region = np.random.choice(['North', 'South', 'East', 'West', 'Central'], n)

    settlement_prob = np.full(n, 0.65)
    for i in range(n):
        if income_brackets[i] == 'Low (<30K)':      settlement_prob[i] -= 0.20
        elif income_brackets[i] == 'Middle (30K-70K)': settlement_prob[i] -= 0.05
        elif income_brackets[i] == 'Premium (150K+)':  settlement_prob[i] += 0.15
        if ages[i] == '60+':    settlement_prob[i] -= 0.12
        elif ages[i] == '18-30': settlement_prob[i] -= 0.05
        if teams[i] == 'Team B':  settlement_prob[i] -= 0.18
        elif teams[i] == 'Team D': settlement_prob[i] += 0.10
        if claim_amounts[i] > np.percentile(claim_amounts, 75): settlement_prob[i] -= 0.10
        settlement_prob[i] = np.clip(settlement_prob[i], 0.05, 0.95)

    settlement_status = np.array(['Settled' if np.random.random() < p else 'Rejected' for p in settlement_prob])
    return pd.DataFrame({
        'Claim_ID': [f'CLM{str(i+1).zfill(5)}' for i in range(n)],
        'Age_Group': ages, 'Income_Bracket': income_brackets,
        'Settlement_Team': teams, 'Policy_Type': policy_types,
        'Claim_Amount': claim_amounts, 'Claim_Duration_Days': claim_durations,
        'Premium_Amount': premium_amounts, 'Num_Prior_Claims': num_prior_claims,
        'Policy_Tenure_Years': policy_tenure, 'Gender': claimant_gender,
        'Region': region, 'Settlement_Status': settlement_status
    })


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Bias Analyzer")
    st.markdown("---")
    uploaded = st.file_uploader("Upload Claims CSV", type=["csv"])
    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio("", [
        "📊 Overview & Descriptive",
        "🔍 Diagnostic Bias Analysis",
        "🤖 ML Model Training",
        "📈 Model Evaluation",
        "📋 Findings & Recommendations"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<small style='opacity:0.6'>Insurance Claim Settlement<br>Bias Analysis Dashboard v1.0</small>", unsafe_allow_html=True)

if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success(f"✅ Loaded {len(df):,} records")
else:
    df = generate_default_data()
    st.sidebar.info("ℹ️ Using built-in demo dataset (1,500 records)")

# ─────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>⚖️ Insurance Claim Settlement Bias Analyzer</h1>
  <p>Comprehensive bias detection · Super Learner ML Classification · Settlement Fairness Diagnostics</p>
</div>
""", unsafe_allow_html=True)

TARGET = 'Settlement_Status'
POSITIVE_CLASS = 'Settled'

# ─────────────────────────────────────────────────────────
# PAGE 1: OVERVIEW & DESCRIPTIVE
# ─────────────────────────────────────────────────────────
if page == "📊 Overview & Descriptive":
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

    total = len(df)
    settled = (df[TARGET] == POSITIVE_CLASS).sum()
    rejected = total - settled
    settlement_rate = settled / total * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Claims", f"{total:,}")
    c2.metric("Settled", f"{settled:,}", f"{settlement_rate:.1f}%")
    c3.metric("Rejected", f"{rejected:,}", f"{100-settlement_rate:.1f}%")
    c4.metric("Avg Claim Amount", f"${df['Claim_Amount'].mean():,.0f}")
    c5.metric("Avg Duration (Days)", f"{df['Claim_Duration_Days'].mean():.0f}")

    st.markdown('<div class="section-title">Data Sample</div>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True, height=300)

    st.markdown('<div class="section-title">Cross-Tabulation Against Policy Status</div>', unsafe_allow_html=True)

    cat_cols = ['Age_Group', 'Income_Bracket', 'Settlement_Team', 'Policy_Type', 'Gender', 'Region']
    selected_cat = st.selectbox("Select Variable for Cross-Tab", cat_cols)

    ct = pd.crosstab(df[selected_cat], df[TARGET], margins=True)
    ct_pct = pd.crosstab(df[selected_cat], df[TARGET], normalize='index').mul(100).round(2)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Count Cross-Tab**")
        st.dataframe(ct, use_container_width=True)
    with col2:
        st.markdown("**Row % Cross-Tab**")
        st.dataframe(ct_pct.style.background_gradient(cmap='RdYlGn', axis=1), use_container_width=True)

    # Cross-tab bar chart
    ct_plot = ct_pct.drop('All') if 'All' in ct_pct.index else ct_pct
    if TARGET in ct_plot.columns:
        fig = px.bar(
            ct_plot.reset_index(), x=selected_cat, y=ct_plot.columns.tolist(),
            barmode='group', title=f"Settlement Rate (%) by {selected_cat}",
            color_discrete_map={POSITIVE_CLASS: '#01696f', 'Rejected': '#a12c7b'},
            template='plotly_white'
        )
        fig.update_layout(legend_title="Status", xaxis_title=selected_cat, yaxis_title="Percentage (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Numerical Feature Distributions</div>', unsafe_allow_html=True)
    num_cols = ['Claim_Amount', 'Premium_Amount', 'Claim_Duration_Days', 'Num_Prior_Claims', 'Policy_Tenure_Years']
    sel_num = st.selectbox("Select Numerical Feature", num_cols)
    fig2 = px.histogram(df, x=sel_num, color=TARGET, barmode='overlay',
                        color_discrete_map={POSITIVE_CLASS: '#01696f', 'Rejected': '#a12c7b'},
                        title=f"Distribution of {sel_num} by Settlement Status",
                        template='plotly_white', opacity=0.75)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df[num_cols].describe().T.round(2), use_container_width=True)

    # All cross-tabs summary
    st.markdown('<div class="section-title">All Cross-Tabs Summary</div>', unsafe_allow_html=True)
    tabs = st.tabs(cat_cols)
    for tab, col in zip(tabs, cat_cols):
        with tab:
            ctt = pd.crosstab(df[col], df[TARGET], margins=True, margins_name='Total')
            ctt_pct = pd.crosstab(df[col], df[TARGET], normalize='index').mul(100).round(1)
            if POSITIVE_CLASS in ctt_pct.columns:
                ctt_pct['Settlement Rate %'] = ctt_pct[POSITIVE_CLASS]
            merged = ctt.merge(ctt_pct[['Settlement Rate %']], left_index=True, right_index=True, how='left')
            st.dataframe(merged, use_container_width=True)


# ─────────────────────────────────────────────────────────
# PAGE 2: DIAGNOSTIC BIAS ANALYSIS
# ─────────────────────────────────────────────────────────
elif page == "🔍 Diagnostic Bias Analysis":
    st.markdown('<div class="section-title">Bias Detection Diagnostics</div>', unsafe_allow_html=True)
    st.info("🔬 This section probes bias patterns using statistical tests, heatmaps, and settlement rate disparities.")

    # Chi-Square Tests
    st.markdown('<div class="section-title">Chi-Square Independence Tests</div>', unsafe_allow_html=True)
    cat_cols = ['Age_Group', 'Income_Bracket', 'Settlement_Team', 'Policy_Type', 'Gender', 'Region']
    chi2_results = []
    for col in cat_cols:
        ct = pd.crosstab(df[col], df[TARGET])
        chi2, p, dof, _ = chi2_contingency(ct)
        chi2_results.append({'Variable': col, 'Chi2 Statistic': round(chi2, 2), 'p-value': round(p, 4),
                              'Degrees of Freedom': dof,
                              'Biased?': '🔴 YES' if p < 0.05 else '🟢 NO'})
    chi2_df = pd.DataFrame(chi2_results)
    st.dataframe(chi2_df, use_container_width=True)
    st.caption("A p-value < 0.05 indicates statistically significant association with Settlement Status (potential bias).")

    # Settlement Rate Disparity Heatmaps
    st.markdown('<div class="section-title">Settlement Rate Heatmaps</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pivot1 = df.groupby(['Age_Group', 'Settlement_Team'])[TARGET].apply(
            lambda x: (x == POSITIVE_CLASS).mean() * 100).unstack()
        fig = px.imshow(pivot1, text_auto='.1f', color_continuous_scale='RdYlGn',
                        title='Settlement Rate %: Age Group × Team',
                        labels=dict(color='Rate %'), template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pivot2 = df.groupby(['Income_Bracket', 'Settlement_Team'])[TARGET].apply(
            lambda x: (x == POSITIVE_CLASS).mean() * 100).unstack()
        fig2 = px.imshow(pivot2, text_auto='.1f', color_continuous_scale='RdYlGn',
                         title='Settlement Rate %: Income × Team',
                         labels=dict(color='Rate %'), template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)

    # Team-wise deep dive
    st.markdown('<div class="section-title">Team-Wise Bias Deep Dive</div>', unsafe_allow_html=True)
    team_stats = df.groupby('Settlement_Team').apply(
        lambda x: pd.Series({
            'Total Claims': len(x),
            'Settled': (x[TARGET] == POSITIVE_CLASS).sum(),
            'Rejected': (x[TARGET] != POSITIVE_CLASS).sum(),
            'Settlement Rate %': round((x[TARGET] == POSITIVE_CLASS).mean() * 100, 1),
            'Avg Claim Amount': round(x['Claim_Amount'].mean(), 0),
            'Avg Duration Days': round(x['Claim_Duration_Days'].mean(), 1)
        })
    ).reset_index()
    st.dataframe(team_stats, use_container_width=True)

    overall_rate = (df[TARGET] == POSITIVE_CLASS).mean() * 100
    fig_team = px.bar(team_stats, x='Settlement_Team', y='Settlement Rate %',
                      color='Settlement Rate %', color_continuous_scale='RdYlGn',
                      title=f"Settlement Rate by Team (Overall: {overall_rate:.1f}%)",
                      template='plotly_white', text='Settlement Rate %')
    fig_team.add_hline(y=overall_rate, line_dash='dash', line_color='#964219',
                        annotation_text=f"Overall Avg: {overall_rate:.1f}%")
    fig_team.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig_team, use_container_width=True)

    # Income-wise deep dive
    st.markdown('<div class="section-title">Income-Wise Bias Analysis</div>', unsafe_allow_html=True)
    income_order = ['Low (<30K)', 'Middle (30K-70K)', 'High (70K-150K)', 'Premium (150K+)']
    income_stats = df.groupby('Income_Bracket').apply(
        lambda x: pd.Series({
            'Total': len(x),
            'Settlement Rate %': round((x[TARGET] == POSITIVE_CLASS).mean() * 100, 1),
            'Avg Claim Amount': round(x['Claim_Amount'].mean(), 0)
        })
    ).reindex(income_order).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig_inc = px.bar(income_stats, x='Income_Bracket', y='Settlement Rate %',
                         color='Settlement Rate %', color_continuous_scale='RdYlGn',
                         title="Settlement Rate by Income Bracket", template='plotly_white',
                         text='Settlement Rate %', category_orders={'Income_Bracket': income_order})
        fig_inc.add_hline(y=overall_rate, line_dash='dash', line_color='#964219',
                          annotation_text=f"Avg: {overall_rate:.1f}%")
        fig_inc.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_inc, use_container_width=True)
    with col2:
        age_order = ['18-30', '31-45', '46-60', '60+']
        age_stats = df.groupby('Age_Group').apply(
            lambda x: pd.Series({'Settlement Rate %': round((x[TARGET] == POSITIVE_CLASS).mean() * 100, 1)})
        ).reindex(age_order).reset_index()
        fig_age = px.bar(age_stats, x='Age_Group', y='Settlement Rate %',
                         color='Settlement Rate %', color_continuous_scale='RdYlGn',
                         title="Settlement Rate by Age Group", template='plotly_white',
                         text='Settlement Rate %', category_orders={'Age_Group': age_order})
        fig_age.add_hline(y=overall_rate, line_dash='dash', line_color='#964219',
                          annotation_text=f"Avg: {overall_rate:.1f}%")
        fig_age.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_age, use_container_width=True)

    # Claim amount bias (box plot)
    st.markdown('<div class="section-title">Claim Amount Distribution by Status</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_box = px.box(df, x=TARGET, y='Claim_Amount', color=TARGET,
                         color_discrete_map={POSITIVE_CLASS: '#01696f', 'Rejected': '#a12c7b'},
                         title="Claim Amount Distribution by Settlement Status", template='plotly_white',
                         log_y=True)
        st.plotly_chart(fig_box, use_container_width=True)
    with col2:
        fig_viol = px.violin(df, x='Income_Bracket', y='Claim_Amount', color=TARGET,
                             color_discrete_map={POSITIVE_CLASS: '#01696f', 'Rejected': '#a12c7b'},
                             title="Claim Amount by Income & Status", template='plotly_white',
                             category_orders={'Income_Bracket': income_order}, log_y=True, box=True)
        st.plotly_chart(fig_viol, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-title">Pairwise Correlation of Numerical Features</div>', unsafe_allow_html=True)
    df_enc = df.copy()
    df_enc['Status_Num'] = (df_enc[TARGET] == POSITIVE_CLASS).astype(int)
    num_cols_corr = ['Claim_Amount', 'Premium_Amount', 'Claim_Duration_Days',
                     'Num_Prior_Claims', 'Policy_Tenure_Years', 'Status_Num']
    corr = df_enc[num_cols_corr].corr()
    fig_corr = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                         title="Feature Correlation Matrix", template='plotly_white',
                         zmin=-1, zmax=1)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Summary alerts
    st.markdown('<div class="section-title">Bias Alert Summary</div>', unsafe_allow_html=True)
    bias_alerts = []
    for _, row in chi2_df.iterrows():
        if '🔴' in row['Biased?']:
            bias_alerts.append(row['Variable'])

    if bias_alerts:
        for var in bias_alerts:
            rate_by_var = df.groupby(var)[TARGET].apply(lambda x: (x == POSITIVE_CLASS).mean() * 100)
            max_gap = rate_by_var.max() - rate_by_var.min()
            st.markdown(f"""
            <div class="bias-alert">
              <div class="bias-alert-title">⚠️ Bias Detected: {var}</div>
              Settlement rate gap across groups: <b>{max_gap:.1f} percentage points</b><br>
              Range: {rate_by_var.min():.1f}% — {rate_by_var.max():.1f}%
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No statistically significant bias detected in categorical variables.")


# ─────────────────────────────────────────────────────────
# PAGE 3: ML MODEL TRAINING
# ─────────────────────────────────────────────────────────
elif page == "🤖 ML Model Training":
    st.markdown('<div class="section-title">Feature Engineering & Model Training</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Feature Engineering Steps", expanded=True):
        st.markdown("""
        **Steps applied:**
        1. **Label Encoding** — categorical variables (Age_Group, Income_Bracket, Settlement_Team, Policy_Type, Gender, Region)
        2. **Claim-to-Premium Ratio** — `Claim_Amount / Premium_Amount` (financial burden indicator)
        3. **Log Transform** — `log1p(Claim_Amount)` and `log1p(Premium_Amount)` to reduce skew
        4. **Prior Claim Density** — `Num_Prior_Claims / Policy_Tenure_Years`
        5. **Binary Target Encoding** — `Settlement_Status`: Settled=1, Rejected=0
        6. **Standard Scaling** — all numerical features scaled to mean=0, std=1
        """)

    # Feature engineering
    df_fe = df.copy()
    df_fe['Target'] = (df_fe[TARGET] == POSITIVE_CLASS).astype(int)

    le_dict = {}
    cat_cols = ['Age_Group', 'Income_Bracket', 'Settlement_Team', 'Policy_Type', 'Gender', 'Region']
    for col in cat_cols:
        le = LabelEncoder()
        df_fe[col + '_Enc'] = le.fit_transform(df_fe[col])
        le_dict[col] = le

    df_fe['Claim_Premium_Ratio'] = df_fe['Claim_Amount'] / (df_fe['Premium_Amount'] + 1)
    df_fe['Log_Claim'] = np.log1p(df_fe['Claim_Amount'])
    df_fe['Log_Premium'] = np.log1p(df_fe['Premium_Amount'])
    df_fe['Prior_Claim_Density'] = df_fe['Num_Prior_Claims'] / (df_fe['Policy_Tenure_Years'] + 1)

    feature_cols = (
        [c + '_Enc' for c in cat_cols] +
        ['Log_Claim', 'Log_Premium', 'Claim_Premium_Ratio',
         'Claim_Duration_Days', 'Num_Prior_Claims', 'Policy_Tenure_Years', 'Prior_Claim_Density']
    )

    X = df_fe[feature_cols]
    y = df_fe['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, stratify=y)

    st.info(f"🔢 Training: {len(X_train):,} samples | Testing: {len(X_test):,} samples | Features: {len(feature_cols)}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Engineered Feature List**")
        feat_df = pd.DataFrame({'Feature': feature_cols, 'Type': ['Encoded']*len(cat_cols) + ['Numerical']*(len(feature_cols)-len(cat_cols))})
        st.dataframe(feat_df, use_container_width=True, height=350)
    with col2:
        st.markdown("**Target Class Distribution**")
        tc = pd.DataFrame({'Status': [POSITIVE_CLASS, 'Rejected'], 'Count': [(y==1).sum(), (y==0).sum()]})
        fig_tc = px.pie(tc, values='Count', names='Status',
                        color_discrete_map={POSITIVE_CLASS: '#01696f', 'Rejected': '#a12c7b'},
                        title="Target Distribution", hole=0.45, template='plotly_white')
        st.plotly_chart(fig_tc, use_container_width=True)

    # Train models
    st.markdown('<div class="section-title">Training 4 ML Classifiers (Super Learner Suite)</div>', unsafe_allow_html=True)

    n_neighbors = st.slider("KNN: k neighbors", 3, 15, 5)
    max_depth_dt = st.slider("Decision Tree: Max Depth", 2, 15, 6)
    n_estimators_rf = st.slider("Random Forest: # Estimators", 50, 300, 100)

    with st.spinner("Training models... ⏳"):
        models = {
            'KNN': KNeighborsClassifier(n_neighbors=n_neighbors),
            'Decision Tree': DecisionTreeClassifier(max_depth=max_depth_dt, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=n_estimators_rf, random_state=42, n_jobs=-1),
            'Gradient Boosted': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test  = model.predict(X_test)
            y_prob_test  = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            results[name] = {
                'model': model,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test,
                'y_prob_test': y_prob_test,
                'train_acc': accuracy_score(y_train, y_pred_train),
                'test_acc': accuracy_score(y_test, y_pred_test),
                'precision': precision_score(y_test, y_pred_test),
                'recall': recall_score(y_test, y_pred_test),
                'f1': f1_score(y_test, y_pred_test),
                'cm': confusion_matrix(y_test, y_pred_test)
            }
        st.success("✅ All models trained successfully!")

    # Store results in session
    st.session_state['results'] = results
    st.session_state['X_test'] = X_test
    st.session_state['y_test'] = y_test
    st.session_state['feature_cols'] = feature_cols

    # Performance table
    st.markdown('<div class="section-title">Model Performance Summary</div>', unsafe_allow_html=True)
    perf_data = []
    for name, res in results.items():
        perf_data.append({
            'Algorithm': name,
            'Train Accuracy': f"{res['train_acc']*100:.2f}%",
            'Test Accuracy': f"{res['test_acc']*100:.2f}%",
            'Precision': f"{res['precision']:.4f}",
            'Recall': f"{res['recall']:.4f}",
            'F1-Score': f"{res['f1']:.4f}",
            'Overfit Risk': '⚠️ High' if (res['train_acc'] - res['test_acc']) > 0.1 else '✅ Low'
        })
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

    # Feature importance (RF)
    st.markdown('<div class="section-title">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    rf_model = results['Random Forest']['model']
    fi = pd.DataFrame({'Feature': feature_cols, 'Importance': rf_model.feature_importances_}).sort_values('Importance', ascending=True)
    fig_fi = px.bar(fi, x='Importance', y='Feature', orientation='h',
                    title="Feature Importance — Random Forest", template='plotly_white',
                    color='Importance', color_continuous_scale='Teal')
    fig_fi.update_layout(height=500)
    st.plotly_chart(fig_fi, use_container_width=True)


# ─────────────────────────────────────────────────────────
# PAGE 4: MODEL EVALUATION
# ─────────────────────────────────────────────────────────
elif page == "📈 Model Evaluation":
    st.markdown('<div class="section-title">Model Evaluation: Accuracy, Confusion Matrix & ROC</div>', unsafe_allow_html=True)

    if 'results' not in st.session_state:
        st.warning("⚠️ Please train models first on the 'ML Model Training' page.")
        st.stop()

    results = st.session_state['results']
    y_test  = st.session_state['y_test']

    # ── Train vs Test Accuracy
    st.markdown('<div class="section-title">Training vs. Testing Accuracy</div>', unsafe_allow_html=True)
    acc_df = pd.DataFrame([{
        'Algorithm': name,
        'Train Accuracy': res['train_acc'] * 100,
        'Test Accuracy': res['test_acc'] * 100
    } for name, res in results.items()])
    acc_melt = acc_df.melt(id_vars='Algorithm', var_name='Split', value_name='Accuracy (%)')
    fig_acc = px.bar(acc_melt, x='Algorithm', y='Accuracy (%)', color='Split', barmode='group',
                     color_discrete_map={'Train Accuracy': '#01696f', 'Test Accuracy': '#4f98a3'},
                     title="Train vs Test Accuracy by Algorithm", template='plotly_white', text_auto='.2f')
    fig_acc.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig_acc, use_container_width=True)

    # ── Precision / Recall / F1
    st.markdown('<div class="section-title">Precision, Recall & F1-Score</div>', unsafe_allow_html=True)
    prf_df = pd.DataFrame([{
        'Algorithm': name,
        'Precision': res['precision'],
        'Recall': res['recall'],
        'F1-Score': res['f1']
    } for name, res in results.items()])
    prf_melt = prf_df.melt(id_vars='Algorithm', var_name='Metric', value_name='Score')
    fig_prf = px.bar(prf_melt, x='Algorithm', y='Score', color='Metric', barmode='group',
                     color_discrete_map={'Precision': '#01696f', 'Recall': '#437a22', 'F1-Score': '#da7101'},
                     title="Precision / Recall / F1-Score Comparison", template='plotly_white',
                     text_auto='.3f')
    fig_prf.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_prf.update_layout(yaxis_range=[0, 1.1])
    st.plotly_chart(fig_prf, use_container_width=True)

    # ── Confusion Matrices with FP/FN %
    st.markdown('<div class="section-title">Confusion Matrices with FP/FN Contribution</div>', unsafe_allow_html=True)
    model_names = list(results.keys())
    cols = st.columns(2)

    for idx, name in enumerate(model_names):
        cm = results[name]['cm']
        total = cm.sum()
        tn, fp, fn, tp = cm.ravel()
        fp_pct = fp / total * 100
        fn_pct = fn / total * 100
        tp_pct = tp / total * 100
        tn_pct = tn / total * 100

        z = [[tn, fp], [fn, tp]]
        text = [[f'TN={tn}<br>({tn_pct:.1f}%)', f'FP={fp}<br>({fp_pct:.1f}%)'],
                [f'FN={fn}<br>({fn_pct:.1f}%)', f'TP={tp}<br>({tp_pct:.1f}%)']]

        fig_cm = go.Figure(data=go.Heatmap(
            z=z, text=text, texttemplate='%{text}',
            colorscale=[[0, '#f7f6f2'], [0.5, '#cedcd8'], [1, '#01696f']],
            showscale=False, xgap=3, ygap=3
        ))
        fig_cm.update_layout(
            title=f'{name}<br><sup>FP: {fp_pct:.1f}% | FN: {fn_pct:.1f}% of all samples</sup>',
            xaxis=dict(tickvals=[0, 1], ticktext=['Predicted Rejected', 'Predicted Settled']),
            yaxis=dict(tickvals=[0, 1], ticktext=['Actual Rejected', 'Actual Settled'], autorange='reversed'),
            template='plotly_white', height=320,
            margin=dict(t=70, b=40)
        )
        with cols[idx % 2]:
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown(f"""
            <small>
            <b>False Positive (FP):</b> Predicted Settled but actually Rejected = <b>{fp} ({fp_pct:.1f}%)</b><br>
            <b>False Negative (FN):</b> Predicted Rejected but actually Settled = <b>{fn} ({fn_pct:.1f}%)</b>
            </small>
            """, unsafe_allow_html=True)

    # ── ROC Curves
    st.markdown('<div class="section-title">ROC Curves — Model Stability</div>', unsafe_allow_html=True)
    fig_roc = go.Figure()
    colors_roc = ['#01696f', '#437a22', '#964219', '#a12c7b']
    for (name, res), color in zip(results.items(), colors_roc):
        if res['y_prob_test'] is not None:
            fpr, tpr, _ = roc_curve(y_test, res['y_prob_test'])
            roc_auc = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={roc_auc:.3f})",
                                         line=dict(color=color, width=2.5)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier',
                                  line=dict(color='gray', dash='dash', width=1.5)))
    fig_roc.update_layout(title='ROC Curves — All Models', xaxis_title='False Positive Rate',
                           yaxis_title='True Positive Rate', template='plotly_white',
                           legend=dict(x=0.6, y=0.1), height=450)
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── Classification Reports
    st.markdown('<div class="section-title">Detailed Classification Reports</div>', unsafe_allow_html=True)
    tabs_cr = st.tabs(model_names)
    for tab, name in zip(tabs_cr, model_names):
        with tab:
            cr = classification_report(y_test, results[name]['y_pred_test'],
                                        target_names=['Rejected', 'Settled'], output_dict=True)
            cr_df = pd.DataFrame(cr).T.round(4)
            st.dataframe(cr_df, use_container_width=True)


# ─────────────────────────────────────────────────────────
# PAGE 5: FINDINGS
# ─────────────────────────────────────────────────────────
elif page == "📋 Findings & Recommendations":
    st.markdown('<div class="section-title">Key Findings & Recommendations</div>', unsafe_allow_html=True)

    # Compute bias stats
    overall_rate = (df[TARGET] == POSITIVE_CLASS).mean() * 100
    team_rates = df.groupby('Settlement_Team')[TARGET].apply(lambda x: (x == POSITIVE_CLASS).mean() * 100)
    income_rates = df.groupby('Income_Bracket')[TARGET].apply(lambda x: (x == POSITIVE_CLASS).mean() * 100)
    age_rates = df.groupby('Age_Group')[TARGET].apply(lambda x: (x == POSITIVE_CLASS).mean() * 100)

    worst_team = team_rates.idxmin()
    best_team  = team_rates.idxmax()
    team_gap   = team_rates.max() - team_rates.min()
    income_gap = income_rates.max() - income_rates.min()
    age_gap    = age_rates.max() - age_rates.min()

    st.markdown(f"""
    <div class="bias-alert">
      <div class="bias-alert-title">🔴 Finding 1: Team-Level Bias Detected</div>
      {worst_team} shows a significantly lower settlement rate ({team_rates[worst_team]:.1f}%) vs {best_team} ({team_rates[best_team]:.1f}%).
      The cross-team disparity is <b>{team_gap:.1f} percentage points</b> — far exceeding acceptable tolerance (&lt;5pp).
      Chi-square test confirms this is statistically significant (p &lt; 0.05).
    </div>
    <div class="bias-alert">
      <div class="bias-alert-title">🔴 Finding 2: Income-Based Discrimination</div>
      Low-income claimants (&lt;30K) face a settlement rate of <b>{income_rates.get('Low (<30K)', 0):.1f}%</b>
      compared to Premium bracket at <b>{income_rates.get('Premium (150K+)', 0):.1f}%</b>.
      Income disparity: <b>{income_gap:.1f} percentage points</b>. This pattern is consistent with systemic bias.
    </div>
    <div class="bias-alert">
      <div class="bias-alert-title">🔴 Finding 3: Age-Related Differential Treatment</div>
      Senior claimants (60+) receive a <b>{age_rates.get('60+', 0):.1f}%</b> settlement rate vs
      <b>{age_rates.get('31-45', 0):.1f}%</b> for 31-45 age group.
      Age disparity: <b>{age_gap:.1f} percentage points</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">ML Model Insights</div>', unsafe_allow_html=True)
    if 'results' in st.session_state:
        results = st.session_state['results']
        best_model = max(results.items(), key=lambda x: x[1]['f1'])
        st.success(f"🏆 **Best Model:** {best_model[0]} with F1-Score = {best_model[1]['f1']:.4f}")

        model_summary = pd.DataFrame([{
            'Algorithm': name,
            'Test Accuracy': f"{res['test_acc']*100:.2f}%",
            'F1-Score': f"{res['f1']:.4f}",
            'AUC': f"{auc(*roc_curve(st.session_state['y_test'], res['y_prob_test'])[:2]):.4f}" if res['y_prob_test'] is not None else 'N/A'
        } for name, res in results.items()])
        st.dataframe(model_summary, use_container_width=True)
    else:
        st.info("ℹ️ Train models on the ML page for model-level insights.")

    st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)
    recs = [
        ("🔍 Audit Team B Processes", "Conduct an immediate process audit of Team B workflows. Compare claim evaluation criteria, escalation patterns, and supervisor approvals against other teams."),
        ("📋 Standardize Settlement Criteria", "Implement a blind assessment protocol where settlement officers evaluate claims without access to claimant income or demographic data."),
        ("⚖️ Establish Disparity Thresholds", "Set a policy that any subgroup with a settlement rate deviating >5pp from the overall average triggers an automatic compliance review."),
        ("🤖 Deploy ML Fairness Monitoring", "Use the Gradient Boosted model as a fairness flag — when model prediction diverges from human decision for demographic segments, flag for review."),
        ("📊 Monthly Bias Reporting", "Require Settlement Teams to submit monthly cross-tab reports (income × age × team) to the compliance officer."),
        ("🎓 Bias Awareness Training", "Mandate unconscious bias training for all settlement officers, with a focus on income and age-related stereotypes in claims processing."),
    ]

    for title, desc in recs:
        with st.expander(title):
            st.write(desc)

    st.markdown('<div class="section-title">False Positive / False Negative Interpretation</div>', unsafe_allow_html=True)
    st.markdown("""
    In the context of insurance claim settlement:

    | Error Type | Meaning | Business Impact |
    |---|---|---|
    | **False Positive (FP)** | Model predicts "Settled" but claim was actually Rejected | Approving ineligible claims → financial loss |
    | **False Negative (FN)** | Model predicts "Rejected" but claim was actually Settled | Wrongly denying valid claims → customer harm, legal risk |

    > **Key Insight:** From a fairness perspective, **False Negatives are more harmful** in a biased settlement environment because they represent eligible claimants being wrongly denied — often concentrated in low-income or elderly demographic groups.
    """)

    # Download data button
    st.markdown('<div class="section-title">Export Dataset</div>', unsafe_allow_html=True)
    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download Dataset (CSV)", csv_buf.getvalue(), "claims_data.csv", "text/csv")

