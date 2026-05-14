import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import (
    ttest_1samp, ttest_ind, ttest_rel, f_oneway, chi2_contingency,
    pearsonr, spearmanr, kendalltau, mannwhitneyu, wilcoxon, kruskal,
    shapiro, normaltest, anderson, levene, bartlett
)

# sklearn imports
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import (
    accuracy_score, r2_score, silhouette_score, mean_absolute_error,
    mean_squared_error, confusion_matrix, classification_report
)
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA

# Supervised learning models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR

# Clustering models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Optional libraries – gracefully handle absence
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# PAGE CONFIG (must be first)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Galactic Data Forge | Senior Data Scientist Workbench",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# DARK THEME (GitHub Dark / VSCode style)
# -------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: #0d1117; }
    .css-1d391kg { background: #161b22; border-right: 1px solid #30363d; }
    .stButton > button { background: #238636; color: white; border: none; border-radius: 6px; font-weight: 500; transition: 0.2s; }
    .stButton > button:hover { background: #2ea043; transform: translateY(-1px); }
    .metric-card { background: #161b22; border-radius: 12px; padding: 1rem; border: 1px solid #30363d; text-align: center; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #ffffff; }
    .metric-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
    .step-header { background: #161b22; border-left: 4px solid #238636; padding: 0.5rem 1rem; margin-bottom: 1.5rem; font-weight: 700; font-size: 1.2rem; color: #ffffff; border-radius: 0 8px 8px 0; }
    .explanation { background: #161b22; border-left: 4px solid #8b949e; padding: 0.75rem 1rem; margin: 1rem 0; font-size: 0.85rem; color: #c9d1d9; border-radius: 8px; }
    .stDataFrame { border-radius: 12px; }
    .stMarkdown, .stText, .stCaption, label, .stSelectbox label, .stMultiSelect label { color: #c9d1d9; }
    hr { border-color: #30363d; margin: 1.2rem 0; }
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div { background-color: #0d1117; border-color: #30363d; color: #c9d1d9; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# SESSION STATE INITIALISATION
# -------------------------------------------------------------------
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'prep_df' not in st.session_state:
    st.session_state.prep_df = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'data_insights' not in st.session_state:
    st.session_state.data_insights = {}
if 'health_log' not in st.session_state:
    st.session_state.health_log = []
if 'ml_model_trained' not in st.session_state:
    st.session_state.ml_model_trained = None
if 'ml_model_name' not in st.session_state:
    st.session_state.ml_model_name = None

for flag in ['analysis_done', 'hypothesis_done', 'preprocessing_done', 'ml_done', 'dl_done', 'prediction_done', 'export_done']:
    if flag not in st.session_state:
        st.session_state[flag] = False

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------
def format_bytes(b):
    for u in ['B', 'KB', 'MB', 'GB']:
        if b < 1024:
            return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} GB"

def health_metrics(df, step_name):
    if df is None:
        return
    rows = df.shape[0]
    cols = df.shape[1]
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / (rows * cols)) * 100 if rows * cols > 0 else 0
    duplicates = df.duplicated().sum()
    dup_pct = (duplicates / rows) * 100 if rows > 0 else 0
    completeness = 100 - missing_pct
    grade = "Excellent" if completeness >= 95 else "Good" if completeness >= 80 else "Poor"
    st.markdown(f"""
    <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0;">
        <div class="metric-card"><div class="metric-value">{rows:,}</div><div class="metric-label">Rows</div></div>
        <div class="metric-card"><div class="metric-value">{cols}</div><div class="metric-label">Columns</div></div>
        <div class="metric-card"><div class="metric-value">{missing_pct:.1f}%</div><div class="metric-label">Missing</div></div>
        <div class="metric-card"><div class="metric-value">{dup_pct:.1f}%</div><div class="metric-label">Duplicates</div></div>
        <div class="metric-card"><div class="metric-value">{completeness:.1f}%</div><div class="metric-label">Completeness</div></div>
        <div class="metric-card"><div class="metric-value">{grade}</div><div class="metric-label">Grade</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.health_log.append(f"**{step_name}** | Rows: {rows}, Missing: {missing_pct:.1f}%, Duplicates: {dup_pct:.1f}%")

def explanation_box(title, content):
    st.markdown(f'<div class="explanation"><strong>📖 {title}</strong><br>{content}</div>', unsafe_allow_html=True)

def detect_data_type(df):
    insights = {}
    insights['rows'] = df.shape[0]
    insights['cols'] = df.shape[1]
    insights['numeric_cols'] = df.select_dtypes(include=[np.number]).shape[1]
    insights['categorical_cols'] = df.select_dtypes(include=['object', 'category']).shape[1]
    insights['datetime_cols'] = df.select_dtypes(include=['datetime64']).shape[1]
    insights['missing_cells'] = df.isnull().sum().sum()
    insights['duplicate_rows'] = df.duplicated().sum()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        skews = df[numeric_cols].skew()
        insights['high_skew'] = skews[abs(skews) > 1].index.tolist()
    else:
        insights['high_skew'] = []
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        high_card = [col for col in cat_cols if df[col].nunique() > 20]
        insights['high_cardinality'] = high_card
    else:
        insights['high_cardinality'] = []
    suggestions = []
    if insights['missing_cells'] > 0:
        suggestions.append("Missing values detected. Consider imputation or dropping columns with >50% missing.")
    if len(insights['high_skew']) > 0:
        suggestions.append(f"Highly skewed columns: {', '.join(insights['high_skew'][:3])}. Consider log transform.")
    if insights['duplicate_rows'] > 0:
        suggestions.append("Duplicate rows found. Consider removing them.")
    if len(insights['high_cardinality']) > 0:
        suggestions.append(f"High cardinality categoricals: {', '.join(insights['high_cardinality'][:2])}. Consider frequency or target encoding.")
    insights['suggestions'] = suggestions
    st.session_state.data_insights = insights
    return insights

def show_data_insights():
    ins = st.session_state.data_insights
    st.markdown("### 🔍 Data Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Numeric Features", ins.get('numeric_cols', 0))
    with col2:
        st.metric("Categorical Features", ins.get('categorical_cols', 0))
    with col3:
        st.metric("Missing Cells", ins.get('missing_cells', 0))
    if ins.get('suggestions'):
        st.warning("**Recommendations:**")
        for s in ins['suggestions']:
            st.write(f"- {s}")

def encode_target(y, problem_type):
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        msg = f"Target encoded: {dict(zip(le.classes_, range(len(le.classes_))))}"
        return y_encoded, msg, le
    return y.astype(float), "Target already numeric", None

def model_health_report(model, X_train, X_test, y_train, y_test, y_pred, problem_type, cv_scores):
    train_score = model.score(X_train, y_train) if hasattr(model, 'score') else None
    test_score = model.score(X_test, y_test) if hasattr(model, 'score') else None
    if train_score is not None and test_score is not None:
        diff = train_score - test_score
        if diff > 0.1:
            overfit_warning = "⚠️ Possible overfitting: Train score much higher than test score."
        elif diff < -0.05:
            overfit_warning = "⚠️ Possible underfitting: Test score higher than train score (unusual)."
        else:
            overfit_warning = "✅ Model is well balanced (no severe overfitting/underfitting)."
    else:
        overfit_warning = "Could not compute train/test scores for this model type."

    st.markdown("### 📋 Model Health Report")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CV Mean Score", f"{cv_scores.mean():.4f}")
        st.metric("CV Std Dev", f"{cv_scores.std():.4f}")
    with col2:
        if train_score is not None:
            st.metric("Train Score", f"{train_score:.4f}")
            st.metric("Test Score", f"{test_score:.4f}")
    st.info(overfit_warning)
    explanation_box("Interpreting Model Health",
                    "**CV Mean** – average performance across folds. **CV Std** – variability; high std may indicate instability. "
                    "Large gap between train and test scores suggests overfitting. Small train score + low test score suggests underfitting.")

# -------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚀 Galactic Data Forge")
    st.markdown("---")
    steps = ["📁 Upload", "🔍 EDA & Insights", "🧹 Clean", "📊 Stats", "🔬 Hypothesis",
             "⚙️ Preprocess", "✨ Feature Select", "🤖 ML Models", "🧠 Deep Learning", "🔮 Predict", "💾 Export"]
    for idx, name in enumerate(steps):
        if st.button(name, key=f"nav_{idx}", use_container_width=True):
            st.session_state.current_step = idx
            st.rerun()
    st.markdown("---")
    if st.session_state.df_raw is not None:
        st.success(f"✅ Dataset ready\n{st.session_state.df_raw.shape[0]} rows\n{st.session_state.df_raw.shape[1]} cols")

# -------------------------------------------------------------------
# STEP 0 – UPLOAD
# -------------------------------------------------------------------
def step_upload():
    st.markdown('<div class="step-header">📁 Step 1: Upload Dataset</div>', unsafe_allow_html=True)
    explanation_box("Why upload?", "Upload CSV, Excel, JSON, or Parquet. Your data stays local. After loading, we automatically analyze data types and provide insights.")
    uploaded = st.file_uploader("Choose a file", type=['csv','xlsx','xls','json','parquet'])
    if uploaded:
        ext = uploaded.name.split('.')[-1].lower()
        try:
            if ext == 'csv':
                df = pd.read_csv(uploaded)
            elif ext in ['xlsx','xls']:
                df = pd.read_excel(uploaded)
            elif ext == 'json':
                df = pd.read_json(uploaded)
            elif ext == 'parquet':
                df = pd.read_parquet(uploaded)
            else:
                st.error("Unsupported")
                return
            st.session_state.df_raw = df.copy()
            st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(20), use_container_width=True)
            detect_data_type(df)
            show_data_insights()
            health_metrics(df, "After Upload")
            if st.button("→ Proceed to EDA", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# -------------------------------------------------------------------
# STEP 1 – EDA & INSIGHTS
# -------------------------------------------------------------------
def step_eda():
    st.markdown('<div class="step-header">🔍 Step 2: Exploratory Data Analysis & Insights</div>', unsafe_allow_html=True)
    df = st.session_state.df_raw.copy()
    detect_data_type(df)
    show_data_insights()
    health_metrics(df, "EDA Start")

    st.subheader("Data Preview (filter columns)")
    cols = st.multiselect("Select columns to display", df.columns, default=df.columns[:5])
    st.dataframe(df[cols].head(100) if cols else df.head(100), use_container_width=True)

    st.subheader("Column Information")
    col_info = pd.DataFrame({
        "Type": df.dtypes,
        "Non‑Null": df.count(),
        "Null %": (df.isnull().sum() / len(df) * 100).round(2),
        "Unique": df.nunique(),
        "Skewness": df.select_dtypes(include=[np.number]).skew().round(2)
    }).fillna("-")
    st.dataframe(col_info, use_container_width=True)
    explanation_box("Interpreting skewness", "Skewness > 1 or < -1 indicates high skew. Log transform may help.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.subheader("Distribution of Numeric Features")
        selected = st.selectbox("Select column", numeric_cols)
        data = df[selected].dropna()
        fig = make_subplots(1, 2, subplot_titles=(f"Histogram of {selected}", f"Boxplot of {selected}"))
        fig.add_trace(go.Histogram(x=data, nbinsx=30, marker_color="#238636"), row=1, col=1)
        fig.add_trace(go.Box(y=data, marker_color="#238636"), row=1, col=2)
        fig.update_layout(height=450, showlegend=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        skew = data.skew()
        kurt = data.kurtosis()
        explanation_box("Statistical interpretation", f"Skewness: {skew:.2f} | Kurtosis: {kurt:.2f}. "
                        f"{'Consider log transform' if abs(skew)>1 else 'Distribution roughly symmetric.'}")

    if len(numeric_cols) >= 2:
        st.subheader("Correlation Matrix")
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Tealgrn", title="Feature Correlations")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        explanation_box("Correlation interpretation", "Values near +1 → strong positive; near -1 → strong negative; near 0 → none. Helps feature selection.")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        st.subheader("Missing Values per Column")
        fig = px.bar(x=missing.index, y=missing.values, title="Missing count", labels={'x':'Column','y':'Missing'})
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("✅ Mark EDA Complete"):
        st.session_state.current_step = 2
        st.rerun()

# -------------------------------------------------------------------
# STEP 2 – CLEAN
# -------------------------------------------------------------------
def step_clean():
    st.markdown('<div class="step-header">🧹 Step 3: Data Cleaning</div>', unsafe_allow_html=True)
    df = st.session_state.df_raw.copy()
    health_metrics(df, "Before Cleaning")
    st.subheader("Cleaning Options")
    col1, col2 = st.columns(2)
    with col1:
        remove_dups = st.checkbox("Remove duplicate rows")
        drop_high_missing = st.checkbox("Drop columns with >50% missing values")
    with col2:
        impute_method = st.selectbox("Numeric imputation", ["None", "Median", "Mean", "KNN (k=5)"])
        fill_cat = st.checkbox("Fill categorical missing with mode")
    cleaned = df.copy()
    if remove_dups:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        st.success(f"Removed {before - len(cleaned)} duplicates")
    if drop_high_missing:
        miss_pct = cleaned.isnull().sum() / len(cleaned) * 100
        cols_drop = miss_pct[miss_pct > 50].index
        if len(cols_drop):
            cleaned = cleaned.drop(columns=cols_drop)
            st.success(f"Dropped columns: {', '.join(cols_drop)}")
    if impute_method != "None":
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        if impute_method == "KNN (k=5)":
            imputer = KNNImputer(n_neighbors=5)
            cleaned[num_cols] = imputer.fit_transform(cleaned[num_cols])
            st.success("KNN imputation applied")
        else:
            for col in num_cols:
                if cleaned[col].isnull().any():
                    if impute_method == "Median":
                        cleaned[col].fillna(cleaned[col].median(), inplace=True)
                    else:
                        cleaned[col].fillna(cleaned[col].mean(), inplace=True)
            st.success(f"Imputed numeric missing with {impute_method}")
    if fill_cat:
        cat_cols = cleaned.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if cleaned[col].isnull().any():
                mode_val = cleaned[col].mode()
                if not mode_val.empty:
                    cleaned[col].fillna(mode_val.iloc[0], inplace=True)
        st.success("Filled categorical missing with mode")
    st.session_state.cleaned_df = cleaned
    st.dataframe(cleaned.head(50), use_container_width=True)
    health_metrics(cleaned, "After Cleaning")
    detect_data_type(cleaned)
    show_data_insights()
    if st.button("✅ Mark Cleaning Complete"):
        st.session_state.current_step = 3
        st.rerun()

# -------------------------------------------------------------------
# STEP 3 – STATISTICAL SUMMARY
# -------------------------------------------------------------------
def step_stats():
    st.markdown('<div class="step-header">📊 Step 4: Statistical Summary</div>', unsafe_allow_html=True)
    df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df_raw
    health_metrics(df, "Statistical Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        st.subheader("Descriptive Statistics")
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        skew_df = df[numeric_cols].skew().to_frame(name="Skewness")
        kurt_df = df[numeric_cols].kurtosis().to_frame(name="Kurtosis")
        extra_stats = pd.concat([skew_df, kurt_df], axis=1)
        st.subheader("Skewness & Kurtosis")
        st.dataframe(extra_stats, use_container_width=True)
        explanation_box("Interpretation", "Skewness > 1: highly right‑skewed; < -1: highly left‑skewed. Kurtosis > 3: heavy tails (leptokurtic).")
    if st.button("✅ Mark Stats Complete"):
        st.session_state.analysis_done = True
        st.session_state.current_step = 4
        st.rerun()

# -------------------------------------------------------------------
# STEP 4 – HYPOTHESIS TESTING
# -------------------------------------------------------------------
def step_hypothesis():
    st.markdown('<div class="step-header">🔬 Step 5: Hypothesis Testing</div>', unsafe_allow_html=True)
    df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df_raw
    alpha = st.select_slider("Significance level (α)", options=[0.01, 0.05, 0.10], value=0.05,
                             format_func=lambda x: f"{x:.0%}")
    st.caption(f"Null hypothesis rejected if p‑value < {alpha:.0%}")
    test_category = st.selectbox("Test Category", [
        "Normality Tests", "One‑Sample Tests", "Two‑Sample (Independent)",
        "Paired Tests", "ANOVA", "Categorical Association", "Correlation", "Variance Equality"
    ])
    if test_category == "Normality Tests":
        col = st.selectbox("Numeric column", df.select_dtypes(include=[np.number]).columns)
        test = st.radio("Test", ["Shapiro-Wilk", "D'Agostino-Pearson", "Anderson-Darling"])
        if st.button("Run"):
            data = df[col].dropna()
            if test == "Shapiro-Wilk":
                stat_val, p = shapiro(data)
                st.metric("W statistic", f"{stat_val:.4f}")
            elif test == "D'Agostino-Pearson":
                stat_val, p = normaltest(data)
                st.metric("K² statistic", f"{stat_val:.4f}")
            else:
                result = anderson(data)
                st.write(f"Anderson-Darling statistic: {result.statistic:.4f}")
                for i, cv in enumerate(result.critical_values):
                    st.write(f"  {result.significance_level[i]}%: {cv:.4f}")
                st.write("Reject normality if statistic > critical value at chosen significance level.")
                return
            st.metric("p-value", f"{p:.4f}")
            st.success("Data appears normal" if p > alpha else "Data does NOT appear normal")
    elif test_category == "One‑Sample Tests":
        col = st.selectbox("Numeric column", df.select_dtypes(include=[np.number]).columns)
        mu = st.number_input("Hypothesized mean", value=0.0)
        test = st.radio("Test", ["t-test", "Wilcoxon signed-rank"])
        if st.button("Run"):
            data = df[col].dropna()
            if test == "t-test":
                stat_val, p = ttest_1samp(data, mu)
                st.metric("t-statistic", f"{stat_val:.4f}")
            else:
                stat_val, p = wilcoxon(data - mu)
                st.metric("W statistic", f"{stat_val:.4f}")
            st.metric("p-value", f"{p:.4f}")
            st.success("Reject H₀: mean ≠ hypothesized" if p < alpha else "Fail to reject H₀")
    elif test_category == "Two‑Sample (Independent)":
        cat_col = st.selectbox("Categorical column (2 groups)", df.select_dtypes(include=['object']).columns)
        num_col = st.selectbox("Numeric column", df.select_dtypes(include=[np.number]).columns)
        test = st.radio("Test", ["t-test (equal var)", "Welch t-test", "Mann-Whitney U"])
        if st.button("Run"):
            groups = df[cat_col].unique()
            if len(groups) != 2:
                st.error("Need exactly 2 groups")
            else:
                g1 = df[df[cat_col]==groups[0]][num_col].dropna()
                g2 = df[df[cat_col]==groups[1]][num_col].dropna()
                if test == "t-test (equal var)":
                    stat_val, p = ttest_ind(g1, g2, equal_var=True)
                    st.metric("t-statistic", f"{stat_val:.4f}")
                elif test == "Welch t-test":
                    stat_val, p = ttest_ind(g1, g2, equal_var=False)
                    st.metric("t-statistic", f"{stat_val:.4f}")
                else:
                    stat_val, p = mannwhitneyu(g1, g2)
                    st.metric("U statistic", f"{stat_val:.4f}")
                st.metric("p-value", f"{p:.4f}")
                st.success("Groups have different means" if p < alpha else "No significant difference")
    elif test_category == "Paired Tests":
        col1 = st.selectbox("First measurement", df.select_dtypes(include=[np.number]).columns)
        col2 = st.selectbox("Second measurement", df.select_dtypes(include=[np.number]).columns)
        test = st.radio("Test", ["Paired t-test", "Wilcoxon signed-rank"])
        if st.button("Run"):
            paired = df[[col1, col2]].dropna()
            if test == "Paired t-test":
                stat_val, p = ttest_rel(paired[col1], paired[col2])
                st.metric("t-statistic", f"{stat_val:.4f}")
            else:
                stat_val, p = wilcoxon(paired[col1] - paired[col2])
                st.metric("W statistic", f"{stat_val:.4f}")
            st.metric("p-value", f"{p:.4f}")
            st.success("Significant difference" if p < alpha else "No significant difference")
    elif test_category == "ANOVA":
        cat_col = st.selectbox("Categorical column (≥2 groups)", df.select_dtypes(include=['object']).columns)
        num_col = st.selectbox("Numeric column", df.select_dtypes(include=[np.number]).columns)
        if st.button("Run ANOVA"):
            groups = [group[num_col].dropna() for name, group in df.groupby(cat_col)]
            if len(groups) >= 2:
                f_stat, p = f_oneway(*groups)
                st.metric("F-statistic", f"{f_stat:.4f}")
                st.metric("p-value", f"{p:.4f}")
                st.success("At least one group mean differs" if p < alpha else "All group means equal")
                if p < alpha and len(groups) <= 5:
                    try:
                        from statsmodels.stats.multicomp import pairwise_tukeyhsd
                        tukey = pairwise_tukeyhsd(df[num_col], df[cat_col])
                        st.subheader("Tukey HSD Post‑hoc")
                        st.dataframe(pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0]))
                    except:
                        pass
    elif test_category == "Categorical Association":
        col1 = st.selectbox("First categorical", df.select_dtypes(include=['object']).columns)
        col2 = st.selectbox("Second categorical", df.select_dtypes(include=['object']).columns)
        test = st.radio("Test", ["Chi-square", "Fisher's exact (2x2 only)"])
        if st.button("Run"):
            contingency = pd.crosstab(df[col1], df[col2])
            if test == "Chi-square":
                chi2, p, dof, ex = chi2_contingency(contingency)
                st.metric("Chi²", f"{chi2:.4f}")
                st.metric("p-value", f"{p:.4f}")
                st.success("Variables are associated" if p < alpha else "No association")
            else:
                from scipy.stats import fisher_exact
                if contingency.shape != (2,2):
                    st.error("Fisher's exact requires 2x2 contingency table")
                else:
                    odds, p = fisher_exact(contingency)
                    st.metric("Odds ratio", f"{odds:.4f}")
                    st.metric("p-value", f"{p:.4f}")
    elif test_category == "Correlation":
        col1 = st.selectbox("Variable 1", df.select_dtypes(include=[np.number]).columns)
        col2 = st.selectbox("Variable 2", df.select_dtypes(include=[np.number]).columns)
        test = st.radio("Method", ["Pearson", "Spearman", "Kendall"])
        if st.button("Run"):
            data = df[[col1, col2]].dropna()
            if test == "Pearson":
                corr, p = pearsonr(data[col1], data[col2])
            elif test == "Spearman":
                corr, p = spearmanr(data[col1], data[col2])
            else:
                corr, p = kendalltau(data[col1], data[col2])
            st.metric("Correlation coefficient", f"{corr:.4f}")
            st.metric("p-value", f"{p:.4f}")
            strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.5 else "weak"
            direction = "positive" if corr > 0 else "negative"
            st.success(f"{direction} {strength} correlation (p={'<' if p<alpha else '>'}0.05)")
    elif test_category == "Variance Equality":
        num_col = st.selectbox("Numeric column", df.select_dtypes(include=[np.number]).columns)
        cat_col = st.selectbox("Grouping column", df.columns)
        test = st.radio("Test", ["Levene", "Bartlett"])
        if st.button("Run"):
            groups = [group[num_col].dropna() for name, group in df.groupby(cat_col)]
            if len(groups) >= 2:
                if test == "Levene":
                    stat_val, p = levene(*groups)
                else:
                    stat_val, p = bartlett(*groups)
                st.metric("Test statistic", f"{stat_val:.4f}")
                st.metric("p-value", f"{p:.4f}")
                st.success("Variances are equal" if p > alpha else "Variances are NOT equal")
    if st.button("✅ Mark Hypothesis Complete"):
        st.session_state.hypothesis_done = True
        st.session_state.current_step = 5
        st.rerun()

# -------------------------------------------------------------------
# STEP 5 – PREPROCESS
# -------------------------------------------------------------------
def step_preprocess():
    st.markdown('<div class="step-header">⚙️ Step 6: Preprocessing (Scaling & Encoding)</div>', unsafe_allow_html=True)
    df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df_raw
    health_metrics(df, "Before Preprocessing")
    st.subheader("Scaling Options")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        scale_method = st.selectbox("Scaler", ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"])
        if scale_method != "None" and st.button("Apply Scaling"):
            scaler = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}[scale_method]
            df_scaled = df.copy()
            df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
            st.session_state.prep_df = df_scaled
            health_metrics(df_scaled, "After Scaling")
            st.success("Scaling applied")
        else:
            st.session_state.prep_df = df.copy()
    else:
        st.session_state.prep_df = df.copy()
        st.info("No numeric columns – scaling skipped")
    st.subheader("Encoding Categoricals")
    cat_cols = st.session_state.prep_df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        encode_cols = st.multiselect("Select categorical columns to encode", cat_cols)
        method = st.radio("Encoding method", ["Label Encoding", "One-Hot Encoding (drop first)", "Target Encoding"])
        if st.button("Apply Encoding"):
            encoded = st.session_state.prep_df.copy()
            if method == "Label Encoding":
                le = LabelEncoder()
                for col in encode_cols:
                    encoded[col] = le.fit_transform(encoded[col].astype(str))
            elif method == "One-Hot Encoding (drop first)":
                encoded = pd.get_dummies(encoded, columns=encode_cols, drop_first=True)
            else:  # Target Encoding
                target = st.selectbox("Target column for target encoding", st.session_state.prep_df.columns)
                if target:
                    for col in encode_cols:
                        mean_encoded = encoded.groupby(col)[target].mean()
                        encoded[col] = encoded[col].map(mean_encoded)
                    st.warning("Target encoding uses mean of target – may cause overfitting without cross‑validation.")
            st.session_state.prep_df = encoded
            st.success("Encoding applied")
            st.dataframe(encoded.head(), use_container_width=True)
            health_metrics(encoded, "After Encoding")
    if st.button("✅ Mark Preprocess Complete"):
        st.session_state.preprocessing_done = True
        st.session_state.current_step = 6
        st.rerun()

# -------------------------------------------------------------------
# STEP 6 – AUTO FEATURE SELECTION
# -------------------------------------------------------------------
def step_feature_selection():
    st.markdown('<div class="step-header">✨ Step 7: Intelligent Feature Selection</div>', unsafe_allow_html=True)
    df = st.session_state.prep_df if st.session_state.prep_df is not None else st.session_state.cleaned_df
    target = st.selectbox("🎯 Target column", df.columns)
    problem = st.radio("Problem type", ["Classification", "Regression"])
    if st.button("Auto‑recommend features"):
        X = df.drop(columns=[target]).select_dtypes(include=[np.number])
        y = df[target].copy()
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            st.info(f"Target encoded: {dict(zip(le.classes_, range(len(le.classes_))))}")
        if X.shape[1] == 0:
            st.error("No numeric features available. Please encode categorical features first (Step 6).")
            return
        X = X.fillna(0)
        y = y.fillna(0).astype(float)
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        try:
            if problem == "Classification":
                mi = mutual_info_classif(X, y, random_state=42)
            else:
                mi = mutual_info_regression(X, y, random_state=42)
        except Exception as e:
            st.error(f"Mutual info failed: {e}. Try selecting a numeric target or check features.")
            return
        mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        corr = X.corrwith(y).abs().sort_values(ascending=False)
        mi_norm = (mi_series - mi_series.min()) / (mi_series.max() - mi_series.min() + 1e-8)
        corr_norm = (corr - corr.min()) / (corr.max() - corr.min() + 1e-8)
        combined = (mi_norm + corr_norm) / 2
        combined = combined.sort_values(ascending=False)
        top_k = min(10, len(combined))
        top_features = combined.head(top_k).index.tolist()
        st.session_state.selected_features = top_features
        fig = px.bar(x=combined.values, y=combined.index, orientation='h',
                     title="Feature Importance (Mutual Info + Correlation)")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"✅ Recommended features: {', '.join(top_features)}")
        st.session_state.selected_features = st.multiselect("Adjust feature set", X.columns.tolist(), default=top_features)
        explanation_box("How it works", "Mutual Information captures non‑linear relationships; Correlation captures linear. Both are normalized and combined.")
    if st.session_state.selected_features:
        st.write(f"**Selected features ({len(st.session_state.selected_features)})**: {', '.join(st.session_state.selected_features)}")
    if st.button("✅ Proceed to Modeling"):
        st.session_state.current_step = 7
        st.rerun()

# -------------------------------------------------------------------
# STEP 7 – CLASSICAL ML (full hyperparameter control)
# -------------------------------------------------------------------
def step_ml():
    st.markdown('<div class="step-header">🤖 Step 8: Classical Machine Learning</div>', unsafe_allow_html=True)
    if not st.session_state.selected_features:
        st.warning("Run Feature Selection first (Step 7).")
        return
    df = st.session_state.prep_df if st.session_state.prep_df is not None else st.session_state.cleaned_df
    target = st.selectbox("Target column", df.columns)
    problem = st.radio("Task", ["Classification", "Regression", "Clustering"])
    with st.expander("⚙️ Hyperparameter Tuning (optional)"):
        use_tuning = st.checkbox("Enable manual hyperparameter tuning", value=False)
        if use_tuning:
            if problem in ["Classification", "Regression"]:
                n_estimators = st.slider("Number of estimators (for tree‑based models)", 10, 500, 100)
                max_depth = st.slider("Max depth (for tree‑based / XGBoost)", 1, 20, 6)
                learning_rate = st.number_input("Learning rate (for boosting)", 0.01, 0.5, 0.1, step=0.01)
                reg_alpha = st.number_input("L1 regularization (alpha)", 0.0, 5.0, 0.0, step=0.1)
                reg_lambda = st.number_input("L2 regularization (lambda)", 0.0, 5.0, 1.0, step=0.1)
            else:
                n_clusters = st.slider("Number of clusters", 2, 20, 3)
    cv_folds = st.slider("Cross‑validation folds", 2, 10, 5)

    if problem == "Classification":
        model_options = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "Gradient Boosting"]
        if XGB_AVAILABLE: model_options.append("XGBoost")
        if LGB_AVAILABLE: model_options.append("LightGBM")
        if CAT_AVAILABLE: model_options.append("CatBoost")
        model_options.append("SVM")
        model_name = st.selectbox("Algorithm", model_options)
    elif problem == "Regression":
        model_options = ["Linear Regression", "Ridge", "Lasso", "ElasticNet", "KNN", "Decision Tree",
                         "Random Forest", "Gradient Boosting"]
        if XGB_AVAILABLE: model_options.append("XGBoost")
        if LGB_AVAILABLE: model_options.append("LightGBM")
        if CAT_AVAILABLE: model_options.append("CatBoost")
        model_options.append("SVR")
        model_name = st.selectbox("Algorithm", model_options)
    else:  # Clustering
        model_options = ["KMeans", "DBSCAN", "Agglomerative", "Gaussian Mixture"]
        model_name = st.selectbox("Algorithm", model_options)

    if st.button(f"🚀 Train & Evaluate {model_name}"):
        X = df[st.session_state.selected_features].copy()
        y = df[target].copy()
        # Encode categorical features in X
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category').cat.codes
        if problem in ["Classification", "Regression"]:
            y_encoded, enc_msg, _ = encode_target(y, problem)
            st.info(enc_msg)
            y = y_encoded
        X = X.fillna(0)
        if problem != "Clustering":
            y = y.fillna(0).astype(float)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        if problem != "Clustering":
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        else:
            X_train = X_scaled

        # Model instantiation
        if problem == "Classification":
            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, C=1/reg_lambda if use_tuning else 1.0)
            elif model_name == "KNN":
                model = KNeighborsClassifier(n_neighbors=max(2, int(n_estimators/10)) if use_tuning else 5)
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier(max_depth=max_depth if use_tuning else None)
            elif model_name == "Random Forest":
                model = RandomForestClassifier(n_estimators=n_estimators if use_tuning else 100, max_depth=max_depth if use_tuning else None)
            elif model_name == "Gradient Boosting":
                model = GradientBoostingClassifier(n_estimators=n_estimators if use_tuning else 100, learning_rate=learning_rate if use_tuning else 0.1, max_depth=max_depth if use_tuning else 3)
            elif model_name == "XGBoost" and XGB_AVAILABLE:
                model = XGBClassifier(n_estimators=n_estimators if use_tuning else 100, max_depth=max_depth if use_tuning else 6, learning_rate=learning_rate if use_tuning else 0.1, reg_alpha=reg_alpha if use_tuning else 0.0, reg_lambda=reg_lambda if use_tuning else 1.0, use_label_encoder=False, eval_metric='logloss')
            elif model_name == "LightGBM" and LGB_AVAILABLE:
                model = LGBMClassifier(n_estimators=n_estimators if use_tuning else 100, max_depth=max_depth if use_tuning else -1, learning_rate=learning_rate if use_tuning else 0.1)
            elif model_name == "CatBoost" and CAT_AVAILABLE:
                model = CatBoostClassifier(iterations=n_estimators if use_tuning else 100, depth=max_depth if use_tuning else 6, learning_rate=learning_rate if use_tuning else 0.1, verbose=0)
            elif model_name == "SVM":
                model = SVC(probability=True, C=reg_lambda if use_tuning else 1.0)
            else:
                st.error("Model not available. Install the required library.")
                return
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='accuracy')
            st.metric("Test Accuracy", f"{acc:.4f}")
            st.metric(f"CV Mean Accuracy ({cv_folds} folds)", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            # Learning curve
            train_sizes, train_scores, test_scores = learning_curve(model, X_scaled, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5))
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_scores.mean(axis=1), mode='lines+markers', name='Train'))
            fig_lc.add_trace(go.Scatter(x=train_sizes, y=test_scores.mean(axis=1), mode='lines+markers', name='Validation'))
            fig_lc.update_layout(title="Learning Curve", xaxis_title="Training examples", yaxis_title="Score", template="plotly_dark")
            st.plotly_chart(fig_lc, use_container_width=True)
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale="Blues")
            fig_cm.update_layout(template="plotly_dark")
            st.plotly_chart(fig_cm, use_container_width=True)
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                imp = pd.Series(model.feature_importances_, index=st.session_state.selected_features).sort_values(ascending=False)
                fig_imp = px.bar(x=imp.values, y=imp.index, orientation='h', title="Feature Importance")
                fig_imp.update_layout(template="plotly_dark")
                st.plotly_chart(fig_imp, use_container_width=True)
            # Model health report
            model_health_report(model, X_train, X_test, y_train, y_test, y_pred, "classification", cv_scores)
            st.session_state.ml_model_trained = model
            st.session_state.ml_model_name = model_name
            st.success(f"{model_name} training complete.")

        elif problem == "Regression":
            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Ridge":
                model = Ridge(alpha=reg_lambda if use_tuning else 1.0)
            elif model_name == "Lasso":
                model = Lasso(alpha=reg_lambda if use_tuning else 1.0)
            elif model_name == "ElasticNet":
                model = ElasticNet(alpha=reg_lambda if use_tuning else 1.0, l1_ratio=0.5)
            elif model_name == "KNN":
                model = KNeighborsRegressor(n_neighbors=max(2, int(n_estimators/10)) if use_tuning else 5)
            elif model_name == "Decision Tree":
                model = DecisionTreeRegressor(max_depth=max_depth if use_tuning else None)
            elif model_name == "Random Forest":
                model = RandomForestRegressor(n_estimators=n_estimators if use_tuning else 100, max_depth=max_depth if use_tuning else None)
            elif model_name == "Gradient Boosting":
                model = GradientBoostingRegressor(n_estimators=n_estimators if use_tuning else 100, learning_rate=learning_rate if use_tuning else 0.1, max_depth=max_depth if use_tuning else 3)
            elif model_name == "XGBoost" and XGB_AVAILABLE:
                model = XGBRegressor(n_estimators=n_estimators if use_tuning else 100, max_depth=max_depth if use_tuning else 6, learning_rate=learning_rate if use_tuning else 0.1, reg_alpha=reg_alpha if use_tuning else 0.0, reg_lambda=reg_lambda if use_tuning else 1.0)
            elif model_name == "LightGBM" and LGB_AVAILABLE:
                model = LGBMRegressor(n_estimators=n_estimators if use_tuning else 100, max_depth=max_depth if use_tuning else -1, learning_rate=learning_rate if use_tuning else 0.1)
            elif model_name == "CatBoost" and CAT_AVAILABLE:
                model = CatBoostRegressor(iterations=n_estimators if use_tuning else 100, depth=max_depth if use_tuning else 6, learning_rate=learning_rate if use_tuning else 0.1, verbose=0)
            elif model_name == "SVR":
                model = SVR(C=reg_lambda if use_tuning else 1.0, epsilon=0.1)
            else:
                st.error("Model not available")
                return
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='r2')
            st.metric("R² Score", f"{r2:.4f}")
            st.metric("MAE", f"{mae:.4f}")
            st.metric("RMSE", f"{rmse:.4f}")
            st.metric(f"CV Mean R² ({cv_folds} folds)", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            # Residual plot
            residuals = y_test - y_pred
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', marker=dict(color='#238636')))
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res.update_layout(title="Residual Plot", xaxis_title="Predicted", yaxis_title="Residuals", template="plotly_dark")
            st.plotly_chart(fig_res, use_container_width=True)
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                imp = pd.Series(model.feature_importances_, index=st.session_state.selected_features).sort_values(ascending=False)
                fig_imp = px.bar(x=imp.values, y=imp.index, orientation='h', title="Feature Importance")
                fig_imp.update_layout(template="plotly_dark")
                st.plotly_chart(fig_imp, use_container_width=True)
            # Model health report
            model_health_report(model, X_train, X_test, y_train, y_test, y_pred, "regression", cv_scores)
            st.session_state.ml_model_trained = model
            st.session_state.ml_model_name = model_name
            st.success(f"{model_name} training complete.")

        else:  # Clustering
            if model_name == "KMeans":
                model = KMeans(n_clusters=n_clusters if use_tuning else 3, random_state=42)
            elif model_name == "DBSCAN":
                eps = st.slider("eps", 0.1, 2.0, 0.5)
                min_samples = st.slider("min_samples", 2, 10, 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
            elif model_name == "Agglomerative":
                model = AgglomerativeClustering(n_clusters=n_clusters if use_tuning else 3)
            else:
                model = GaussianMixture(n_components=n_clusters if use_tuning else 3, random_state=42)
            labels = model.fit_predict(X_scaled)
            if len(set(labels)) > 1:
                sil = silhouette_score(X_scaled, labels)
                st.metric("Silhouette Score", f"{sil:.4f}")
            else:
                st.warning("Only one cluster formed. Adjust parameters.")
            # PCA visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig_pca = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=labels.astype(str), title="PCA Projection of Clusters")
            fig_pca.update_layout(template="plotly_dark")
            st.plotly_chart(fig_pca, use_container_width=True)
            st.success("Clustering complete.")

    if st.button("✅ Mark ML Complete"):
        st.session_state.ml_done = True
        st.session_state.current_step = 8
        st.rerun()

# -------------------------------------------------------------------
# STEP 8 – DEEP LEARNING (only if TensorFlow available)
# -------------------------------------------------------------------
def step_deep_learning():
    st.markdown('<div class="step-header">🧠 Step 9: Deep Learning</div>', unsafe_allow_html=True)
    if not st.session_state.selected_features:
        st.warning("Run Feature Selection first.")
        return
    if not TF_AVAILABLE:
        st.error("TensorFlow not installed. Run `pip install tensorflow` to enable deep learning.")
        if st.button("✅ Mark Deep Learning Complete (skip)"):
            st.session_state.dl_done = True
            st.session_state.current_step = 9
            st.rerun()
        return

    df = st.session_state.prep_df if st.session_state.prep_df is not None else st.session_state.cleaned_df
    target = st.selectbox("Target column", df.columns)
    y_raw = df[target]
    
    # Intelligent task detection
    if y_raw.dtype in ['int64', 'float64'] and y_raw.nunique() > 20:
        default_task = "Regression"
        st.info(f"Target `{target}` has {y_raw.nunique()} unique numeric values → using Regression.")
    elif y_raw.dtype == 'object' or y_raw.nunique() <= 20:
        default_task = "Classification"
        st.info(f"Target `{target}` has {y_raw.nunique()} unique categories → using Classification.")
    else:
        default_task = "Regression"
    
    problem = st.radio("Task", ["Classification", "Regression"], index=0 if default_task == "Classification" else 1)
    
    # Warn if classification but many unique values
    if problem == "Classification" and y_raw.nunique() > 20:
        st.error(f"❌ Target `{target}` has {y_raw.nunique()} unique values. Classification with >20 classes is not recommended. Please switch to Regression.")
        return
    
    model_type = st.selectbox("Architecture", ["ANN", "CNN 1D", "RNN (LSTM)", "Transformer"])
    epochs = st.slider("Max epochs", 10, 200, 50)
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
    validation_split = st.slider("Validation split", 0.1, 0.3, 0.2)
    early_stop_patience = st.slider("Early stopping patience", 3, 20, 10)
    explanation_box("Preventing overfitting", "Early stopping monitors validation loss and restores best weights. Dropout and batch normalization are included.")

    if st.button(f"🚀 Train {model_type}"):
        X = df[st.session_state.selected_features].copy()
        y = df[target].copy()
        
        # Encode categorical features in X
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category').cat.codes
        
        # Handle target encoding
        le = None
        if problem == "Classification":
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                st.info(f"Target encoded: {dict(zip(le.classes_, range(len(le.classes_))))}")
            num_classes = len(np.unique(y))
            if num_classes <= 2:
                output_units = 1
                output_activation = 'sigmoid'
                loss = 'binary_crossentropy'
                metric = 'accuracy'
            else:
                output_units = num_classes
                output_activation = 'softmax'
                loss = 'sparse_categorical_crossentropy'
                metric = 'accuracy'
        else:  # Regression
            # Ensure y is float
            if y.dtype == 'object':
                # Try to convert to numeric, if fail then error
                y = pd.to_numeric(y, errors='coerce')
                if y.isnull().any():
                    st.error("Regression target contains non‑numeric values. Please encode or use Classification.")
                    return
            output_units = 1
            output_activation = 'linear'
            loss = 'mse'
            metric = 'mae'
        
        # Fill missing values
        X = X.fillna(0)
        if problem == "Regression" or problem == "Classification":
            y = y.fillna(0).astype(float)
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Reshape for CNN/RNN
        if model_type in ["CNN 1D", "RNN (LSTM)"]:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            input_shape = (X_train.shape[1], 1)
        else:
            input_shape = (X_train.shape[1],)
        
        # Build model
        model = keras.Sequential()
        if model_type == "ANN":
            model.add(layers.Input(shape=input_shape))
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dropout(0.3))
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dropout(0.3))
            model.add(layers.Dense(32, activation='relu'))
        elif model_type == "CNN 1D":
            model.add(layers.Input(shape=input_shape))
            model.add(layers.Conv1D(64, 2, activation='relu'))
            model.add(layers.MaxPooling1D(2))
            model.add(layers.Conv1D(32, 2, activation='relu'))
            model.add(layers.GlobalAveragePooling1D())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dropout(0.3))
        elif model_type == "RNN (LSTM)":
            model.add(layers.Input(shape=input_shape))
            model.add(layers.LSTM(64, return_sequences=True))
            model.add(layers.LSTM(32))
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dropout(0.3))
        else:  # Transformer
            model.add(layers.Input(shape=input_shape))
            model.add(layers.Dense(64))
            model.add(layers.MultiHeadAttention(num_heads=4, key_dim=32))
            model.add(layers.GlobalAveragePooling1D())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(output_units, activation=output_activation))
        model.compile(optimizer='adam', loss=loss, metrics=[metric])
        
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=early_stop_patience, restore_best_weights=True)
        with st.spinner(f"Training {model_type}..."):
            history = model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size,
                                callbacks=[early_stop], verbose=0)
        
        # Plot training curves
        fig = make_subplots(1, 2, subplot_titles=('Loss', metric.upper()))
        fig.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss'), row=1, col=1)
        fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss'), row=1, col=1)
        fig.add_trace(go.Scatter(y=history.history[metric], name=f'Train {metric}'), row=1, col=2)
        fig.add_trace(go.Scatter(y=history.history[f'val_{metric}'], name=f'Val {metric}'), row=1, col=2)
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Evaluation
        if problem == "Classification":
            _, acc = model.evaluate(X_test, y_test, verbose=0)
            st.metric("Test Accuracy", f"{acc:.4f}")
            y_pred_prob = model.predict(X_test)
            if output_units == 1:
                y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            else:
                y_pred = np.argmax(y_pred_prob, axis=1)
            st.subheader("Classification Report")
            from sklearn.metrics import classification_report
            report = classification_report(y_test.astype(int), y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
            cm = confusion_matrix(y_test.astype(int), y_pred)
            fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale="Blues")
            fig_cm.update_layout(template="plotly_dark")
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            _, mae = model.evaluate(X_test, y_test, verbose=0)
            st.metric("Test MAE", f"{mae:.4f}")
            y_pred = model.predict(X_test).flatten()
            residuals = y_test - y_pred
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', marker=dict(color='#238636')))
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res.update_layout(title="Residual Plot", xaxis_title="Predicted", yaxis_title="Residuals", template="plotly_dark")
            st.plotly_chart(fig_res, use_container_width=True)
            r2 = r2_score(y_test, y_pred)
            st.metric("R² Score", f"{r2:.4f}")
        
        st.success("Deep learning training complete!")
    
    if st.button("✅ Mark Deep Learning Complete"):
        st.session_state.dl_done = True
        st.session_state.current_step = 9
        st.rerun()
# -------------------------------------------------------------------
# STEP 9 – PREDICT (placeholder)
# -------------------------------------------------------------------
def step_predict():
    st.markdown('<div class="step-header">🔮 Step 10: Batch Prediction</div>', unsafe_allow_html=True)
    st.info("Upload a new file (same features) to get predictions from a trained model. Model persistence not yet implemented – you can extend this.")
    if st.button("✅ Mark Predict Complete"):
        st.session_state.prediction_done = True
        st.session_state.current_step = 10
        st.rerun()

# -------------------------------------------------------------------
# STEP 10 – EXPORT
# -------------------------------------------------------------------
def step_export():
    st.markdown('<div class="step-header">💾 Step 11: Export Processed Data</div>', unsafe_allow_html=True)
    df = st.session_state.prep_df if st.session_state.prep_df is not None else st.session_state.cleaned_df
    st.subheader("Download Processed Dataset")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "processed_data.csv", "text/csv")
    if st.button("✅ Mark Export Complete"):
        st.session_state.export_done = True
        st.rerun()

# -------------------------------------------------------------------
# MAIN DISPATCH
# -------------------------------------------------------------------
def main():
    if st.session_state.df_raw is None:
        step_upload()
    else:
        step_map = {
            0: step_upload, 1: step_eda, 2: step_clean, 3: step_stats,
            4: step_hypothesis, 5: step_preprocess, 6: step_feature_selection,
            7: step_ml, 8: step_deep_learning, 9: step_predict, 10: step_export
        }
        current = st.session_state.current_step
        if current in step_map:
            step_map[current]()
        else:
            st.error("Invalid step")

if __name__ == "__main__":
    main()