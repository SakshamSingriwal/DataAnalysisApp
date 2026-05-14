import streamlit as st
import pandas as pd
import numpy as np
from utils import compute_quality, fmt_bytes
from file_handler import read_any_file
from stats_tests import run_hypothesis_testing
from preprocessing import run_preprocessing
from ml_models import run_ml_models
from deep_learning import run_deep_learning

# ════════════════════════════════════════════════════════════════
# CONFIGURATION & SETUP
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Data Analytics & ML Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background-color: #060810; color: #f1f5f9; }

    .stApp { background: linear-gradient(135deg, #060810 0%, #0a0b14 100%); }

    .css-1d391kg { background-color: #060810 !important; }

    .stSidebar { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
                 border-right: 1px solid #334155; }

    .stTextInput, .stNumberInput, .stSelectbox, .stMultiselect, .stTextArea {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3) !important;
    }

    .css-1offfwp { background-color: #0f172a !important; }

    .stDataFrame { border-radius: 8px !important; }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }

    .stExpander { background-color: #0f172a !important;
                  border: 1px solid #334155 !important;
                  border-radius: 8px !important; }

    .stTabs [data-baseweb="tab-list"] {
        background-color: #0f172a !important;
        border-radius: 8px !important;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #94a3b8 !important;
        border-radius: 6px !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
    }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    .metric-card {
        background: linear-gradient(145deg, #0f172a, #1e293b);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .metric-value { font-size: 2em; font-weight: 800; color: #6366f1; }
    .metric-label { color: #94a3b8; font-size: 0.9em; margin-top: 8px; }

    .step-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        margin: 20px 0;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .section-card {
        background: linear-gradient(145deg, #0f172a, #1e293b);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'prep_df' not in st.session_state:
    st.session_state.prep_df = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# Initialize completion flags
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'hypothesis_done' not in st.session_state:
    st.session_state.hypothesis_done = False
if 'preprocessing_done' not in st.session_state:
    st.session_state.preprocessing_done = False
if 'ml_done' not in st.session_state:
    st.session_state.ml_done = False
if 'dl_done' not in st.session_state:
    st.session_state.dl_done = False
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
if 'export_done' not in st.session_state:
    st.session_state.export_done = False

# Initialize cleaning options state
if 'clean_remove_duplicates' not in st.session_state:
    st.session_state.clean_remove_duplicates = False
if 'clean_drop_missing_cols' not in st.session_state:
    st.session_state.clean_drop_missing_cols = False
if 'clean_impute_method' not in st.session_state:
    st.session_state.clean_impute_method = 'Median'
if 'clean_fill_categorical_mode' not in st.session_state:
    st.session_state.clean_fill_categorical_mode = False

# ════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #6366f1; font-size: 1.8em; margin: 0;'>📊</h1>
        <h2 style='color: #f1f5f9; font-size: 1.2em; margin: 8px 0; font-weight: 700;'>Data Analytics</h2>
        <p style='color: #94a3b8; font-size: 0.9em; margin: 0;'>ML Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation steps
    steps = [
        ("📂 Upload", "Upload your dataset"),
        ("🔍 Explore", "Data exploration"),
        ("🧹 Clean", "Data cleaning"),
        ("📈 Analyse", "Statistical analysis"),
        ("🧪 Hypotheses", "Hypothesis testing"),
        ("⚙️ Preprocess", "Data preprocessing"),
        ("🤖 ML Models", "Machine learning"),
        ("🧠 Deep Learning", "Neural networks"),
        ("🔮 Predict", "Make predictions"),
        ("💾 Export", "Export results")
    ]

    for i, (icon_text, desc) in enumerate(steps):
        if st.button(f"{icon_text} {desc}", key=f"nav_{i}",
                    use_container_width=True,
                    help=f"Go to {desc}"):
            st.session_state.current_step = i
            st.rerun()

    st.markdown("---")

    # Progress indicator
    try:
        completed_steps = sum([
            st.session_state.get('df_raw') is not None,
            st.session_state.get('cleaned_df') is not None,
            st.session_state.get('prep_df') is not None,
            st.session_state.get('analysis_done', False),
            st.session_state.get('hypothesis_done', False),
            st.session_state.get('preprocessing_done', False),
            st.session_state.get('ml_done', False),
            st.session_state.get('dl_done', False),
            st.session_state.get('prediction_done', False),
            st.session_state.get('export_done', False)
        ])

        st.progress(completed_steps / len(steps))
        st.caption(f"Progress: {completed_steps}/{len(steps)} steps completed")
    except Exception as e:
        st.progress(0.0)
        st.caption("Progress: Error calculating progress")

# ════════════════════════════════════════════════════════════════
# MAIN APPLICATION LOGIC
# ════════════════════════════════════════════════════════════════

def main():
    try:
        # Define step functions
        step_functions = [
            step_upload,
            step_explore,
            step_clean,
            step_analyse,
            step_hypotheses,
            step_preprocess,
            step_ml_models,
            step_deep_learning,
            step_predict,
            step_export
        ]

        # Allow upload step even when no data is loaded
        if st.session_state.df_raw is None and st.session_state.current_step == 0:
            step_upload()
        # Show landing page when no data is loaded and not on upload step
        elif st.session_state.df_raw is None:
            show_landing_page()
        # Execute current step when data is loaded
        elif st.session_state.current_step < len(step_functions):
            step_functions[st.session_state.current_step]()
        else:
            st.error("Invalid step number")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")

def show_landing_page():
    try:
        st.markdown("""
        <div style='text-align: center; padding: 60px 20px;'>
            <h1 style='color: #f1f5f9; font-size: 3em; margin-bottom: 20px; font-weight: 800;'>
                Welcome to Data Analytics & ML Platform
            </h1>
            <p style='color: #94a3b8; font-size: 1.2em; margin-bottom: 40px; max-width: 600px; margin-left: auto; margin-right: auto;'>
                Professional-grade data analysis and machine learning platform with support for 15+ file formats,
                comprehensive statistical testing, and advanced ML models.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Feature showcase
        col1, col2, col3 = st.columns(3)
        features = [
            ("📊", "Data Exploration", "Interactive visualizations and statistical summaries"),
            ("🤖", "Machine Learning", "Classification, regression, and clustering models"),
            ("🧠", "Deep Learning", "Neural network builder with custom architectures"),
            ("📈", "Statistical Tests", "12 categories of hypothesis testing"),
            ("⚙️", "Preprocessing", "Encoding, scaling, imputation, and feature engineering"),
            ("💾", "Export Ready", "Save models, predictions, and processed data")
        ]

        for i, (icon, title, desc) in enumerate(features):
            col = [col1, col2, col3][i % 3]
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 2em; margin-bottom: 10px;'>{icon}</div>
                    <div style='font-weight: 600; color: #f1f5f9;'>{title}</div>
                    <div style='color: #94a3b8; font-size: 0.8em; margin-top: 8px;'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("👈 **Start by uploading your dataset using the sidebar navigation**")
    except Exception as e:
        st.error(f"Error displaying landing page: {str(e)}")

# ════════════════════════════════════════════════════════════════
# STEP FUNCTIONS
# ════════════════════════════════════════════════════════════════

def step_upload():
    st.markdown('<div class="step-header">📂 Step 1: Upload Your Dataset</div>',
                unsafe_allow_html=True)

    st.markdown("""
    Upload your data file in any of the supported formats. The platform automatically detects
    the format and loads your data for analysis.
    """)

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'db', 'sql', 'py', 'pdf',
              'docx', 'jpg', 'jpeg', 'png', 'gif', 'pbix', 'xml', 'ods', 'feather'],
        help="Supported formats: CSV, Excel, JSON, Parquet, SQLite, SQL, Python, PDF, Word, Images, Power BI, XML, ODS, Feather"
    )

    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            try:
                df, meta, error_code = read_any_file(uploaded_file)
                
                if df is None:
                    st.error(f"❌ Error loading file: {error_code}")
                    if meta.get('trace'):
                        st.error(f"Details: {meta['trace']}")
                    if meta.get('notes'):
                        for note in meta['notes']:
                            st.warning(note)
                    return
                
                st.session_state.df_raw = df
                st.session_state.current_step = 1
                st.success(f"✅ Successfully loaded {len(df)} rows × {len(df.columns)} columns")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")

def step_explore():
    if st.session_state.df_raw is None:
        st.warning("Please upload a dataset first")
        return

    st.markdown('<div class="step-header">🔍 Step 2: Explore Your Data</div>',
                unsafe_allow_html=True)

    df = st.session_state.df_raw

    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    with col4:
        try:
            memory_usage = df.memory_usage(deep=True).sum()
            st.metric("Memory Usage", fmt_bytes(memory_usage))
        except Exception as e:
            st.metric("Memory Usage", "N/A")

    # Data preview
    st.subheader("📋 Data Preview")
    try:
        st.dataframe(df.head(100), use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying data preview: {str(e)}")

    # Column info
    st.subheader("📊 Column Information")
    try:
        col_info = pd.DataFrame({
            'Type': df.dtypes,
            'Non-Null': df.count(),
            'Null': df.isnull().sum(),
            'Unique': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying column information: {str(e)}")

    # Quick stats for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.subheader("📈 Quick Statistics")
        try:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying statistics: {str(e)}")

    if st.button("✅ Mark as Explored"):
        st.session_state.current_step = 2
        st.rerun()

def step_clean():
    if st.session_state.df_raw is None:
        st.warning("Please upload a dataset first")
        return

    st.markdown('<div class="step-header">🧹 Step 3: Clean Your Data</div>',
                unsafe_allow_html=True)

    df = st.session_state.df_raw.copy()

    # Data quality assessment before cleaning
    try:
        original_score, original_grade, original_missing_pct, original_dup_pct = compute_quality(df)
        st.markdown("**Current raw dataset quality**")
        st.metric("Data Quality Score", f"{original_score:.1f}%")
        st.caption(f"Grade: {original_grade} | Missing: {original_missing_pct:.1f}% | Duplicates: {original_dup_pct:.1f}%")
    except Exception as e:
        st.warning(f"Could not compute quality score: {str(e)}")
        st.metric("Data Quality Score", "N/A")

    # Cleaning options
    st.subheader("🛠️ Cleaning Options")

    col1, col2 = st.columns(2)

    with col1:
        remove_duplicates = st.checkbox(
            "Remove duplicate rows",
            value=st.session_state.get('clean_remove_duplicates', False),
            key='clean_remove_duplicates'
        )

        drop_missing_cols = st.checkbox(
            "Drop columns with >50% missing values",
            value=st.session_state.get('clean_drop_missing_cols', False),
            key='clean_drop_missing_cols'
        )

    with col2:
        imputation_method = st.selectbox(
            "Numeric imputation strategy",
            ["None", "Median", "Regression"],
            index=["None", "Median", "Regression"].index(
                st.session_state.get('clean_impute_method', 'Median')
            ),
            key='clean_impute_method'
        )

        fill_categorical_mode = st.checkbox(
            "Fill missing categorical values with mode",
            value=st.session_state.get('clean_fill_categorical_mode', False),
            key='clean_fill_categorical_mode'
        )

    # Apply cleaning operations
    cleaned_df = df.copy()

    try:
        if remove_duplicates:
            original_len = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            removed_count = original_len - len(cleaned_df)
            if removed_count > 0:
                st.success(f"✅ Removed {removed_count} duplicate rows")

        if drop_missing_cols:
            missing_pct = cleaned_df.isnull().sum() / len(cleaned_df) * 100
            cols_to_drop = missing_pct[missing_pct > 50].index
            if len(cols_to_drop) > 0:
                cleaned_df = cleaned_df.drop(columns=cols_to_drop)
                st.success(f"✅ Dropped {len(cols_to_drop)} columns: {', '.join(cols_to_drop)}")

        if imputation_method == "Median":
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            filled_count = 0
            for col in numeric_cols:
                if cleaned_df[col].isnull().sum() > 0:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                    filled_count += 1
            if filled_count > 0:
                st.success(f"✅ Filled missing values in {filled_count} numeric columns with median")

        elif imputation_method == "Regression":
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            target_cols = [c for c in numeric_cols if cleaned_df[c].isnull().sum() > 0]
            if len(target_cols) == 0:
                st.info("No numeric column with missing values found for regression imputation.")
            else:
                regression_target = st.selectbox(
                    "Target column for regression imputation",
                    target_cols,
                    key='clean_regression_target'
                )
                predictor_cols = [c for c in numeric_cols if c != regression_target]

                if len(predictor_cols) == 0:
                    st.warning("Regression requires at least one numeric predictor column.")
                else:
                    regression_features = st.multiselect(
                        "Predictor columns",
                        predictor_cols,
                        default=predictor_cols,
                        key='clean_regression_features'
                    )

                    if len(regression_features) == 0:
                        st.warning("Select at least one predictor column to run regression imputation.")
                    else:
                        if st.button("Run regression imputation"):
                            try:
                                from sklearn.linear_model import LinearRegression

                                train_df = cleaned_df.dropna(subset=[regression_target] + regression_features)
                                predict_df = cleaned_df[
                                    cleaned_df[regression_target].isnull() &
                                    cleaned_df[regression_features].notnull().all(axis=1)
                                ]

                                if len(train_df) < 5:
                                    st.warning("Not enough rows to train a regression model. Need at least 5 complete rows.")
                                elif len(predict_df) == 0:
                                    st.info("No rows found that can be imputed with the selected predictors.")
                                else:
                                    model = LinearRegression()
                                    model.fit(train_df[regression_features], train_df[regression_target])
                                    preds = model.predict(predict_df[regression_features])
                                    cleaned_df.loc[predict_df.index, regression_target] = preds
                                    st.success(f"✅ Regression imputed {len(preds)} missing values in '{regression_target}'")
                            except Exception as e:
                                st.error(f"Regression imputation error: {e}")

        if fill_categorical_mode:
            cat_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
            filled_count = 0
            for col in cat_cols:
                if cleaned_df[col].isnull().sum() > 0:
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
                        filled_count += 1
            if filled_count > 0:
                st.success(f"✅ Filled missing values in {filled_count} categorical columns with mode")
    except Exception as e:
        st.error(f"❌ Error during cleaning operations: {str(e)}")
        cleaned_df = df.copy()  # Revert to original if error occurs

    # Manual column selection
    st.subheader("🎯 Column Selection")
    selected_cols = st.multiselect(
        "Select columns to keep",
        cleaned_df.columns.tolist(),
        default=cleaned_df.columns.tolist(),
        key='selected_columns'
    )

    if selected_cols:
        cleaned_df = cleaned_df[selected_cols]

    # Compute quality after cleaning operations
    try:
        updated_score, updated_grade, updated_missing_pct, updated_dup_pct = compute_quality(cleaned_df)
        st.markdown("**Post-clean quality summary**")
        st.metric("Updated Data Quality Score", f"{updated_score:.1f}%")
        st.caption(f"Grade: {updated_grade} | Missing: {updated_missing_pct:.1f}% | Duplicates: {updated_dup_pct:.1f}%")
    except Exception as e:
        st.warning(f"Could not compute updated quality score: {str(e)}")

    # Real-time missing details
    missing_by_col = cleaned_df.isnull().sum()
    missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)

    if not missing_by_col.empty:
        st.subheader("🔍 Missing Value Breakdown")
        missing_summary = pd.DataFrame({
            'Missing Count': missing_by_col,
            'Missing %': (missing_by_col / len(cleaned_df) * 100).round(2)
        })
        st.dataframe(missing_summary, use_container_width=True)

        st.info(f"{len(missing_by_col)} column(s) currently contain missing values.")

        if st.checkbox("Show rows with missing values", value=False, key='show_missing_rows'):
            rows_with_missing = cleaned_df[cleaned_df.isnull().any(axis=1)]
            st.write(f"Showing up to 20 rows with missing data ({len(rows_with_missing)} total rows affected)")
            st.dataframe(rows_with_missing.head(20), use_container_width=True)
    else:
        st.success("No missing values remain in the cleaned dataset.")

    st.session_state.cleaned_df = cleaned_df

    st.subheader("📋 Cleaned Data Preview")
    st.dataframe(cleaned_df.head(50), use_container_width=True)

    if st.button("✅ Mark as Cleaned"):
        st.session_state.current_step = 3
        st.rerun()

def step_analyse():
    if st.session_state.cleaned_df is None:
        st.warning("Please clean your data first")
        return

    st.markdown('<div class="step-header">📈 Step 4: Statistical Analysis</div>',
                unsafe_allow_html=True)

    df = st.session_state.cleaned_df

    # Summary statistics
    st.subheader("📊 Summary Statistics")

    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

        if len(cat_cols) > 0:
            st.subheader("📋 Categorical Variables")
            for col in cat_cols:
                try:
                    st.write(f"**{col}**")
                    value_counts = df[col].value_counts().head(10)
                    st.dataframe(value_counts, use_container_width=True)
                except Exception as e:
                    st.error(f"Error analyzing column {col}: {str(e)}")

        # Correlation analysis
        if len(numeric_cols) > 1:
            st.subheader("🔗 Correlation Matrix")
            try:
                corr = df[numeric_cols].corr()
                st.dataframe(corr.style.background_gradient(cmap='coolwarm'), use_container_width=True)
            except Exception as e:
                st.error(f"Error computing correlation matrix: {str(e)}")
    except Exception as e:
        st.error(f"Error during statistical analysis: {str(e)}")

    if st.button("✅ Mark as Analysed"):
        st.session_state.analysis_done = True
        st.session_state.current_step = 4
        st.rerun()

def step_hypotheses():
    if st.session_state.cleaned_df is None:
        st.warning("Please clean your data first")
        return

    st.markdown('<div class="step-header">🧪 Step 5: Hypothesis Testing</div>',
                unsafe_allow_html=True)

    try:
        run_hypothesis_testing(st.session_state.cleaned_df)
    except Exception as e:
        st.error(f"Error during hypothesis testing: {str(e)}")

    if st.button("✅ Mark as Tested"):
        st.session_state.hypothesis_done = True
        st.session_state.current_step = 5
        st.rerun()

def step_preprocess():
    if st.session_state.cleaned_df is None:
        st.warning("Please clean your data first")
        return

    st.markdown('<div class="step-header">⚙️ Step 6: Data Preprocessing</div>',
                unsafe_allow_html=True)

    try:
        st.session_state.prep_df = run_preprocessing(st.session_state.cleaned_df)
    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        st.session_state.prep_df = None

    if st.button("✅ Mark as Preprocessed"):
        st.session_state.preprocessing_done = True
        st.session_state.current_step = 6
        st.rerun()

def step_ml_models():
    if st.session_state.prep_df is None:
        st.warning("Please preprocess your data first")
        return

    st.markdown('<div class="step-header">🤖 Step 7: Machine Learning Models</div>',
                unsafe_allow_html=True)

    try:
        run_ml_models(st.session_state.prep_df)
    except Exception as e:
        st.error(f"Error during machine learning: {str(e)}")

    if st.button("✅ Mark as Trained"):
        st.session_state.ml_done = True
        st.session_state.current_step = 7
        st.rerun()

def step_deep_learning():
    if st.session_state.prep_df is None:
        st.warning("Please preprocess your data first")
        return

    st.markdown('<div class="step-header">🧠 Step 8: Deep Learning</div>',
                unsafe_allow_html=True)

    try:
        run_deep_learning(st.session_state.prep_df)
    except Exception as e:
        st.error(f"Error during deep learning: {str(e)}")

    if st.button("✅ Mark as Built"):
        st.session_state.dl_done = True
        st.session_state.current_step = 8
        st.rerun()

def step_predict():
    if st.session_state.prep_df is None:
        st.warning("Please preprocess your data first")
        return

    st.markdown('<div class="step-header">🔮 Step 9: Make Predictions</div>',
                unsafe_allow_html=True)

    st.info("Prediction functionality would be implemented here based on trained models")

    if st.button("✅ Mark as Predicted"):
        st.session_state.prediction_done = True
        st.session_state.current_step = 9
        st.rerun()

def step_export():
    st.markdown('<div class="step-header">💾 Step 10: Export Results</div>',
                unsafe_allow_html=True)

    st.info("Export functionality would be implemented here to save models, predictions, and processed data")

    if st.button("✅ Mark as Exported"):
        st.session_state.export_done = True
        st.session_state.current_step = 10
        st.rerun()

if __name__ == "__main__":
    main()