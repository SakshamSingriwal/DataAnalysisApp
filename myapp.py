import pandas as pd
import numpy as np
import streamlit as st

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DataIQ — Professional Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Full professional CSS ─────────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}
    .main {background:#07090f;}
    .block-container {padding:1.2rem 2rem 2rem 2rem;}

    /* Metric cards */
    [data-testid="metric-container"] {
        background:#111827;
        border:1px solid #1f2937;
        border-radius:12px;
        padding:14px 18px;
        transition:border-color .2s;
    }
    [data-testid="metric-container"]:hover {border-color:#6366f1;}
    [data-testid="metric-label"] {color:#9ca3af !important;font-size:.78em !important;}
    [data-testid="metric-value"] {color:#f9fafb !important;font-size:1.5em !important;font-weight:700 !important;}
    [data-testid="metric-delta"] {font-size:.78em !important;}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background:#111827;border-radius:12px;padding:6px;gap:4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius:8px;color:#6b7280;padding:8px 20px;font-weight:500;font-size:.88em;
    }
    .stTabs [aria-selected="true"] {
        background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
        color:white !important;
    }

    /* Expanders */
    .stExpander {
        border:1px solid #1f2937 !important;
        border-radius:12px !important;
        background:#111827 !important;
    }
    summary {color:#e5e7eb !important;font-weight:600 !important;}

    /* Buttons */
    .stButton>button {
        background:linear-gradient(135deg,#6366f1,#8b5cf6);
        color:white;border:none;border-radius:10px;
        padding:10px 24px;font-weight:600;font-size:.9em;
        width:100%;transition:opacity .2s,transform .1s;
    }
    .stButton>button:hover {opacity:.88;transform:translateY(-1px);}
    .stButton>button:active {transform:translateY(0);}

    /* Selectbox / multiselect */
    .stSelectbox>div>div,.stMultiSelect>div>div {
        background:#111827;border:1px solid #1f2937;border-radius:8px;color:#e5e7eb;
    }

    /* DataFrames */
    div[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;}
    div[data-testid="stDataFrame"] table {background:#111827;}

    /* Scrollbar */
    ::-webkit-scrollbar {width:6px;height:6px;}
    ::-webkit-scrollbar-track {background:#07090f;}
    ::-webkit-scrollbar-thumb {background:#374151;border-radius:3px;}

    /* Custom boxes */
    .iq-info {
        background:#0f172a;border-left:3px solid #6366f1;
        padding:12px 16px;border-radius:0 10px 10px 0;
        color:#c7d2fe;font-size:.88em;line-height:1.8;margin:8px 0;
    }
    .iq-success {
        background:#052e16;border-left:3px solid #22c55e;
        padding:12px 16px;border-radius:0 10px 10px 0;
        color:#86efac;font-size:.88em;line-height:1.8;margin:8px 0;
    }
    .iq-warn {
        background:#1c1208;border-left:3px solid #f59e0b;
        padding:12px 16px;border-radius:0 10px 10px 0;
        color:#fcd34d;font-size:.88em;line-height:1.8;margin:8px 0;
    }
    .iq-danger {
        background:#1c0a0a;border-left:3px solid #ef4444;
        padding:12px 16px;border-radius:0 10px 10px 0;
        color:#fca5a5;font-size:.88em;line-height:1.8;margin:8px 0;
    }
    .iq-section {
        background:#111827;border:1px solid #1f2937;
        border-radius:14px;padding:20px 24px;margin:18px 0 10px 0;
    }
    .iq-header {
        background:linear-gradient(135deg,#6366f1,#8b5cf6,#ec4899);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        font-size:1.25em;font-weight:800;margin-bottom:4px;
    }
    .iq-badge-green  {background:#14532d;color:#86efac;padding:2px 10px;border-radius:20px;font-size:.78em;font-weight:600;}
    .iq-badge-yellow {background:#713f12;color:#fcd34d;padding:2px 10px;border-radius:20px;font-size:.78em;font-weight:600;}
    .iq-badge-red    {background:#7f1d1d;color:#fca5a5;padding:2px 10px;border-radius:20px;font-size:.78em;font-weight:600;}
    .iq-badge-orange {background:#431407;color:#fb923c;padding:2px 10px;border-radius:20px;font-size:.78em;font-weight:600;}
    .iq-divider {border:none;border-top:1px solid #1f2937;margin:16px 0;}
    .iq-kv {display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1f2937;font-size:.88em;}
    .iq-kv-k {color:#9ca3af;}
    .iq-kv-v {color:#f9fafb;font-weight:600;}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def ibox(t):  st.markdown(f'<div class="iq-info">💡 {t}</div>',    unsafe_allow_html=True)
def sbox(t):  st.markdown(f'<div class="iq-success">✅ {t}</div>', unsafe_allow_html=True)
def wbox(t):  st.markdown(f'<div class="iq-warn">⚠️ {t}</div>',   unsafe_allow_html=True)
def dbox(t):  st.markdown(f'<div class="iq-danger">🔴 {t}</div>',  unsafe_allow_html=True)
def divider(): st.markdown('<hr class="iq-divider">', unsafe_allow_html=True)
def section(icon, title):
    st.markdown(f'<div class="iq-header">{icon} &nbsp;{title}</div>', unsafe_allow_html=True)

def safe_skew(series):
    # Safely compute skewness — converts to numeric first to avoid string dtype crash
    try:
        num = pd.to_numeric(series, errors='coerce').dropna()
        return float(num.skew()) if len(num) >= 3 else 0.0
    except Exception:
        return 0.0

def safe_mean(series):
    # Safely compute mean with numeric coercion
    try:
        return float(pd.to_numeric(series, errors='coerce').dropna().mean())
    except Exception:
        return 0.0

def safe_median(series):
    # Safely compute median with numeric coercion
    try:
        return float(pd.to_numeric(series, errors='coerce').dropna().median())
    except Exception:
        return 0.0

def severity_badge(pct):
    # Return coloured HTML badge based on missing percentage
    if pct < 5:
        return '<span class="iq-badge-green">🟢 Low</span>'
    elif pct < 20:
        return '<span class="iq-badge-yellow">🟡 Medium</span>'
    elif pct < 50:
        return '<span class="iq-badge-orange">🟠 High</span>'
    else:
        return '<span class="iq-badge-red">🔴 Critical</span>'

def get_recommendation(col, df):
    # Return best strategy string based on dtype, missing %, skewness
    present = df[col].dropna()
    pct     = df[col].isnull().sum() / max(len(df), 1) * 100
    dtype   = str(df[col].dtype)

    if len(present) == 0:
        return "Drop column — no data present at all"
    if pct < 5:
        return f"Drop rows — only {pct:.1f}% missing, negligible data loss"
    if dtype == 'object' or str(df[col].dtype).startswith('string'):
        mode = present.mode()
        mv   = mode[0] if len(mode) > 0 else 'N/A'
        return f'Fill with Mode → "{mv}" (text column, mean/median not applicable)'

    # Numeric path — use skewness to decide mean vs median
    num     = pd.to_numeric(present, errors='coerce').dropna()
    if len(num) == 0:
        mode = present.mode()
        mv   = mode[0] if len(mode) > 0 else 'N/A'
        return f'Fill with Mode → "{mv}" (could not parse as numeric)'
    skw     = safe_skew(num)
    if abs(skw) > 1:
        return f"Fill with Median → {safe_median(num):.4f} (skewed {skw:.2f}, outliers present)"
    return f"Fill with Mean → {safe_mean(num):.4f} (symmetric {skw:.2f}, no major outliers)"

def quality_score(df):
    # Calculate 0-100 data quality score — safe against empty DataFrames
    if df is None or len(df) == 0 or df.shape[1] == 0:
        return 0, '🔴 No Data'
    tc   = df.shape[0] * df.shape[1]
    mp   = float(df.isnull().sum().sum()) / tc * 100 if tc > 0 else 0.0
    dp   = float(df.duplicated().sum()) / len(df) * 100
    mp   = 0.0 if np.isnan(mp) else mp
    dp   = 0.0 if np.isnan(dp) else dp
    sc   = max(0, round(100 - min(mp*2, 40) - min(dp*2, 30)))
    gr   = '🟢 Excellent' if sc>=90 else '🟡 Good' if sc>=70 else '🟠 Fair' if sc>=50 else '🔴 Poor'
    return sc, gr


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 12px;'>
        <div style='font-size:3em;'>📊</div>
        <div style='font-size:1.1em;font-weight:800;color:#f9fafb;margin-top:6px;letter-spacing:.5px;'>DataIQ</div>
        <div style='color:#6b7280;font-size:.78em;margin-top:3px;'>Professional Analytics Suite</div>
    </div>
    """, unsafe_allow_html=True)
    divider()
    st.markdown("""
    <div style='color:#9ca3af;font-size:.83em;line-height:2.3;padding:0 4px;'>
        <b style='color:#e5e7eb;font-size:.9em;'>WORKFLOW</b><br>
        <span style='color:#6366f1;'>①</span> &nbsp;Upload Dataset<br>
        <span style='color:#6366f1;'>②</span> &nbsp;Explore Structure<br>
        <span style='color:#6366f1;'>③</span> &nbsp;Value Counts<br>
        <span style='color:#6366f1;'>④</span> &nbsp;GroupBy Analysis<br>
        <span style='color:#6366f1;'>⑤</span> &nbsp;Health &amp; Clean<br>
        <span style='color:#6366f1;'>⑥</span> &nbsp;Auto Insights
    </div>
    """, unsafe_allow_html=True)
    divider()

    # Reset button — always visible once a file is loaded
    if st.session_state.get('file_loaded'):
        if st.button("🔄 Reset to Original", key='sidebar_reset'):
            for k in ['cleaned_df','file_name','file_loaded']:
                st.session_state.pop(k, None)
            st.rerun()

    st.markdown("<div style='color:#374151;font-size:.72em;text-align:center;margin-top:20px;'>DataIQ v2.0 · Streamlit + Pandas</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding:24px 0 4px;'>
    <h1 style='
        background:linear-gradient(135deg,#6366f1,#8b5cf6,#ec4899);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        font-size:2.6em;font-weight:900;margin:0;letter-spacing:-1px;'>
        DataIQ Analytics
    </h1>
    <p style='color:#4b5563;margin-top:6px;font-size:.95em;letter-spacing:.3px;'>
        Upload &nbsp;·&nbsp; Explore &nbsp;·&nbsp; Clean &nbsp;·&nbsp; Analyse &nbsp;·&nbsp; Insight
    </p>
</div>
""", unsafe_allow_html=True)
divider()


# ════════════════════════════════════════════════════════════════
# SECTION 1 — FILE UPLOAD
# ════════════════════════════════════════════════════════════════
section("📂", "Upload Your Dataset")
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
ibox("Supported formats: <b>CSV</b> and <b>Excel (.xlsx / .xls)</b>. Any size, any domain. Upload and the app does the rest.")

_, mid, _ = st.columns([1, 2, 1])
with mid:
    file = st.file_uploader(" ", type=['csv','xlsx','xls'], label_visibility='collapsed')

if file is None:
    # Landing cards when no file uploaded
    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1,'🔍','Deep Explore','Stats, types, distributions at a glance'),
        (c2,'🧹','Smart Clean','Auto-diagnose and fix with one click'),
        (c3,'🔗','Correlations','Discover hidden column relationships'),
        (c4,'📋','Auto Report','Instant professional summary report'),
    ]:
        col.markdown(f"""
        <div style='background:#111827;border:1px solid #1f2937;border-radius:12px;
                    padding:20px;text-align:center;border-top:3px solid #6366f1;'>
            <div style='font-size:2em;'>{icon}</div>
            <div style='color:#f9fafb;font-weight:700;margin-top:8px;'>{title}</div>
            <div style='color:#6b7280;font-size:.82em;margin-top:4px;'>{desc}</div>
        </div>""", unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════════════
# READ FILE — robust error handling for all edge cases
# ════════════════════════════════════════════════════════════════
try:
    if file.name.lower().endswith('.csv'):
        # Try UTF-8 first then fall back to latin-1 for special characters
        try:
            data = pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            data = pd.read_csv(file, encoding='latin-1')
    else:
        data = pd.read_excel(file)

    # Reject completely empty files
    if data.empty or data.shape[1] == 0:
        dbox("The uploaded file is empty or has no columns. Please upload a valid dataset.")
        st.stop()

    # Strip whitespace from column names to prevent hidden key errors
    data.columns = data.columns.astype(str).str.strip()

    # Drop columns that are 100% empty — they carry zero information
    all_null_cols = [c for c in data.columns if data[c].isnull().all()]
    if all_null_cols:
        data = data.drop(columns=all_null_cols)
        wbox(f"Auto-removed {len(all_null_cols)} fully-empty column(s): <b>{all_null_cols}</b>")

except Exception as e:
    dbox(f"Could not read file: <b>{e}</b>. Please check the file and try again.")
    st.stop()


# Initialise session state — only reset when a new file is uploaded
if st.session_state.get('file_name') != file.name:
    st.session_state.cleaned_df = data.copy()
    st.session_state.file_name  = file.name
    st.session_state.file_loaded = True

# Always pull working copy from session state
df = st.session_state.cleaned_df

sbox(f"<b>{file.name}</b> loaded — {data.shape[0]:,} rows × {data.shape[1]} columns")
st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# SECTION 2 — EXPLORE
# ════════════════════════════════════════════════════════════════
divider()
section("🔍", "Step 1 — Explore Dataset")
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# Top KPI row
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Rows",           f"{data.shape[0]:,}")
k2.metric("Columns",        f"{data.shape[1]}")
k3.metric("Numeric Cols",   f"{len(data.select_dtypes(include='number').columns)}")
k4.metric("Text Cols",      f"{len(data.select_dtypes(include='object').columns)}")
k5.metric("Missing Cells",  f"{data.isnull().sum().sum():,}")
k6.metric("Duplicate Rows", f"{data.duplicated().sum():,}")

st.markdown("<br>", unsafe_allow_html=True)

t1,t2,t3,t4,t5 = st.tabs(["📋 Raw Data","📊 Statistics","🔢 Types","❓ Missing","🔁 Duplicates"])

# Raw preview with adjustable row count
with t1:
    rows_to_show = st.slider("Rows to preview", 5, min(1000, len(data)), min(25, len(data)), key='exp_slider')
    st.dataframe(data.head(rows_to_show), use_container_width=True)

# Statistical summary — numeric and categorical separately
with t2:
    num_df = data.select_dtypes(include='number')
    cat_df = data.select_dtypes(include='object')
    if not num_df.empty:
        st.markdown("**Numeric Summary**")
        try:
            st.dataframe(
                num_df.describe().T.style.background_gradient(cmap='Blues').format(precision=3),
                use_container_width=True
            )
        except Exception:
            st.dataframe(num_df.describe().T, use_container_width=True)
    else:
        wbox("No numeric columns found.")
    if not cat_df.empty:
        st.markdown("**Categorical Summary**")
        st.dataframe(cat_df.describe().T, use_container_width=True)

# Data type table with sample values
with t3:
    dtype_rows = []
    for col in data.columns:
        sample = data[col].dropna()
        dtype_rows.append({
            'Column'         : col,
            'Dtype'          : str(data[col].dtype),
            'Non-Null Count' : int(data[col].notnull().sum()),
            'Null Count'     : int(data[col].isnull().sum()),
            'Unique Values'  : int(data[col].nunique()),
            'Sample Value'   : str(sample.iloc[0]) if len(sample) > 0 else 'N/A'
        })
    st.dataframe(pd.DataFrame(dtype_rows), use_container_width=True)

# Missing values heatmap-style table
with t4:
    miss_tab = pd.DataFrame({
        'Column'  : data.columns,
        'Missing' : data.isnull().sum().values,
        'Present' : data.notnull().sum().values,
        '%'       : (data.isnull().sum().values / max(len(data),1) * 100).round(2)
    }).query('Missing > 0').sort_values('%', ascending=False)
    if miss_tab.empty:
        sbox("Zero missing values in the raw uploaded dataset.")
    else:
        try:
            st.dataframe(
                miss_tab.style.background_gradient(subset=['%'], cmap='Reds').format({'%':'{:.2f}%'}),
                use_container_width=True
            )
        except Exception:
            st.dataframe(miss_tab, use_container_width=True)

# Duplicate rows preview
with t5:
    dup_tab = data[data.duplicated(keep=False)]
    if dup_tab.empty:
        sbox("Zero duplicate rows in the raw uploaded dataset.")
    else:
        wbox(f"Found <b>{data.duplicated().sum():,}</b> duplicate rows ({data.duplicated().sum()/len(data)*100:.1f}% of total)")
        st.dataframe(dup_tab.sort_values(by=data.columns.tolist()).head(100), use_container_width=True)


# ════════════════════════════════════════════════════════════════
# SECTION 3 — VALUE COUNTS (no row limit — shows all unique values)
# ════════════════════════════════════════════════════════════════
divider()
section("🔢", "Step 2 — Column Value Counts")
ibox("See every unique value in any column and its exact frequency. No limit — full picture.")

with st.expander("Open Value Count Explorer"):
    vc_col = st.selectbox("Select column", options=list(data.columns), key='vc_col')

    if st.button("Show All Unique Values", key='vc_btn'):
        try:
            vc = data[vc_col].value_counts(dropna=False).reset_index()
            vc.columns = ['Value', 'Count']
            vc['Value'] = vc['Value'].astype(str)
            vc['%']     = (vc['Count'] / max(len(data),1) * 100).round(2)

            v1,v2,v3,v4 = st.columns(4)
            v1.metric("Total Rows",     f"{len(data):,}")
            v2.metric("Unique Values",  f"{data[vc_col].nunique():,}")
            v3.metric("Missing",        f"{data[vc_col].isnull().sum():,}")
            v4.metric("Top Value",      str(vc.iloc[0]['Value']) if len(vc)>0 else 'N/A')

            try:
                st.dataframe(
                    vc.style.background_gradient(subset=['Count'], cmap='Blues').format({'%':'{:.2f}%'}),
                    use_container_width=True,
                    height=min(600, 60 + 36*len(vc))
                )
            except Exception:
                st.dataframe(vc, use_container_width=True)

            sbox(f"Showing all <b>{len(vc)}</b> unique values in <b>{vc_col}</b>")
        except Exception as e:
            dbox(f"Value count error: {e}")


# ════════════════════════════════════════════════════════════════
# SECTION 4 — GROUPBY
# ════════════════════════════════════════════════════════════════
divider()
section("📦", "Step 3 — GroupBy Analysis")
ibox("Group your data by one or more columns and apply aggregations to summarise patterns.")

with st.expander("Open GroupBy Builder"):
    gb1,gb2,gb3 = st.columns(3)
    with gb1:
        grp_cols = st.multiselect("Group By", options=list(data.columns), key='grp_cols')
    with gb2:
        agg_col  = st.selectbox("Aggregate On", options=list(data.columns), key='agg_col')
    with gb3:
        agg_op   = st.selectbox("Operation", ['sum','mean','max','min','count','median','std','var'], key='agg_op')

    if grp_cols:
        try:
            gb_res = data.groupby(grp_cols, observed=True).agg(Result=(agg_col, agg_op)).reset_index()
            g1,g2,g3 = st.columns(3)
            g1.metric("Groups Found",    f"{len(gb_res):,}")
            g2.metric("Operation",       agg_op.upper())
            g3.metric("Column Analysed", agg_col)
            st.dataframe(gb_res, use_container_width=True)
            sbox(f"<b>{agg_op.upper()}</b> of <b>{agg_col}</b> across <b>{', '.join(grp_cols)}</b>")
        except Exception as e:
            dbox(f"GroupBy failed: <b>{e}</b>")
    else:
        ibox("Select at least one column to group by.")


# ════════════════════════════════════════════════════════════════
# SECTION 5 — HEALTH CHECK + INLINE CLEANING
# Every issue is shown WITH its fix button at the same place
# ════════════════════════════════════════════════════════════════
divider()
section("🧹", "Step 4 — Data Health Check & Inline Cleaning")
ibox("Every issue is diagnosed and fixed <b>at the same point</b>. No need to scroll. Clean column by column or all at once.")

# Always refresh df from session state
df = st.session_state.cleaned_df

# Guard: if somehow empty, offer reset
if len(df) == 0 or df.shape[1] == 0:
    dbox("The working dataset is empty. Please reset.")
    if st.button("🔄 Reset Dataset", key='empty_reset'):
        st.session_state.cleaned_df = data.copy()
        st.rerun()
    st.stop()

# Live dashboard — recalculates after every action
def render_health(df, data):
    qs, gr = quality_score(df)
    h1,h2,h3,h4,h5,h6 = st.columns(6)
    h1.metric("Rows",           f"{df.shape[0]:,}", delta=f"{df.shape[0]-data.shape[0]:,}")
    h2.metric("Columns",        f"{df.shape[1]}",   delta=f"{df.shape[1]-data.shape[1]}")
    h3.metric("Missing Cells",  f"{df.isnull().sum().sum():,}")
    h4.metric("Duplicates",     f"{df.duplicated().sum():,}")
    h5.metric("Quality Score",  f"{qs}/100")
    h6.metric("Grade",          gr)

render_health(df, data)
st.markdown("<br>", unsafe_allow_html=True)

# ── 5A: DUPLICATE ROWS ───────────────────────────────────────
st.markdown('<div class="iq-section">', unsafe_allow_html=True)
st.markdown("#### 🔁 Duplicate Row Diagnosis")

df = st.session_state.cleaned_df
dup_mask  = df.duplicated(keep=False)
dup_extra = df.duplicated(keep='first').sum()

if dup_extra == 0:
    sbox("No duplicate rows. Dataset is unique.")
else:
    dup_pct = dup_extra / max(len(df),1) * 100
    da,db,dc = st.columns(3)
    da.metric("Extra Copies",    f"{dup_extra:,}")
    db.metric("Rows Involved",   f"{int(dup_mask.sum()):,}")
    dc.metric("Duplication %",   f"{dup_pct:.2f}%")

    if dup_pct < 1:      sbox(f"LOW — {dup_pct:.2f}%. Safe and clean to remove.")
    elif dup_pct < 5:    wbox(f"MEDIUM — {dup_pct:.2f}%. Remove before analysis.")
    elif dup_pct < 20:   wbox(f"HIGH — {dup_pct:.2f}%. Must remove before ML training.")
    else:                dbox(f"CRITICAL — {dup_pct:.2f}%. Data pipeline may have a bug.")

    # Show exactly which rows are duplicated
    with st.expander("🔬 View All Duplicated Rows"):
        st.dataframe(df[dup_mask].sort_values(by=df.columns.tolist()), use_container_width=True)
        st.markdown("**Frequency of each duplicate group:**")
        try:
            freq = df[dup_mask].groupby(df.columns.tolist(), observed=True).size().reset_index(name='Copies')
            st.dataframe(freq.sort_values('Copies', ascending=False), use_container_width=True)
        except Exception as e:
            wbox(f"Could not build frequency table: {e}")

    ibox(f"""<b>Recommendation → Remove Duplicates</b><br>
    {dup_extra:,} extra rows removed → {len(df)-dup_extra:,} rows remain.<br>
    Duplicates inflate statistics and bias ML models.<br>
    ⚠️ Verify: if the same event genuinely happened twice (e.g. same customer, two orders), do NOT remove.""")

    # FIX BUTTON — right next to the diagnosis
    if st.button("🗑️ Remove All Duplicate Rows", key='fix_dupes'):
        st.session_state.cleaned_df = st.session_state.cleaned_df.drop_duplicates()
        sbox(f"Removed {dup_extra:,} duplicate rows.")
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)


# ── 5B: MISSING VALUES — COLUMN BY COLUMN WITH INLINE FIX ────
st.markdown('<div class="iq-section">', unsafe_allow_html=True)
st.markdown("#### ❓ Missing Value Diagnosis & Inline Fix")

df = st.session_state.cleaned_df
mc = df.isnull().sum()
missing_cols = mc[mc > 0].index.tolist()

if not missing_cols:
    sbox("No missing values in any column. Dataset is complete.")
else:
    total_miss = mc.sum()
    total_cell = df.shape[0] * df.shape[1]
    ma,mb,mc_ = st.columns(3)
    ma.metric("Columns Affected",    f"{len(missing_cols)} / {df.shape[1]}")
    mb.metric("Total Missing Cells", f"{total_miss:,}")
    mc_.metric("Overall Missing %",  f"{total_miss/max(total_cell,1)*100:.2f}%")

    divider()

    # Summary diagnosis table
    diag_data = []
    for col in missing_cols:
        cnt = int(df[col].isnull().sum())
        pct = cnt / max(len(df),1) * 100
        diag_data.append({
            'Column'         : col,
            'Type'           : str(df[col].dtype),
            'Missing'        : cnt,
            'Missing %'      : f"{pct:.2f}%",
            'Severity'       : '🟢 Low' if pct<5 else '🟡 Medium' if pct<20 else '🟠 High' if pct<50 else '🔴 Critical',
            'Recommendation' : get_recommendation(col, df)
        })
    st.dataframe(pd.DataFrame(diag_data), use_container_width=True)

    divider()
    st.markdown("##### 🔬 Fix Each Column — Inline")

    # Per column expander with full diagnosis + fix in the same place
    for col in missing_cols:
        cnt  = int(df[col].isnull().sum())
        pct  = cnt / max(len(df),1) * 100
        dtype = str(df[col].dtype)
        pres  = df[col].dropna()
        rec   = get_recommendation(col, df)

        with st.expander(f"🔎  {col}   ·   {cnt:,} missing   ·   {pct:.1f}%   ·   {dtype}"):

            # Severity and recommendation
            if pct < 5:      sbox(f"LOW severity — {pct:.1f}% missing.")
            elif pct < 20:   wbox(f"MEDIUM severity — {pct:.1f}% missing.")
            elif pct < 50:   wbox(f"HIGH severity — {pct:.1f}% missing.")
            else:            dbox(f"CRITICAL — {pct:.1f}% missing. Consider dropping column.")

            ibox(f"<b>Recommended Strategy:</b> {rec}")

            # Stats for numeric columns
            if dtype != 'object':
                num_pres = pd.to_numeric(pres, errors='coerce').dropna()
                if len(num_pres) >= 1:
                    s1,s2,s3,s4,s5 = st.columns(5)
                    s1.metric("Mean",    f"{num_pres.mean():.3f}")
                    s2.metric("Median",  f"{num_pres.median():.3f}")
                    s3.metric("Std",     f"{num_pres.std():.3f}")
                    s4.metric("Min",     f"{num_pres.min():.3f}")
                    s5.metric("Max",     f"{num_pres.max():.3f}")

            # Show which rows have missing values in this column
            with st.expander(f"  📄 View rows missing {col}"):
                null_rows = df[df[col].isnull()]
                st.dataframe(null_rows.head(50), use_container_width=True)
                if len(null_rows) > 50:
                    st.caption(f"Showing 50 of {len(null_rows)} null rows")

            divider()
            st.markdown(f"**Apply fix for `{col}`:**")

            # Strategy picker + apply — all in same expander
            fix_key = f"fix_{col}"
            strat   = st.selectbox(
                "Strategy",
                ['Drop rows', 'Fill Mean', 'Fill Median', 'Fill Mode', 'Fill Custom'],
                key=f"strat_{col}"
            )

            cval = None
            if strat == 'Fill Custom':
                cval = st.text_input("Custom value", placeholder="e.g. 0 / Unknown / N/A", key=f"cv_{col}")

            # Apply button — fixes only this column immediately
            if st.button(f"✅ Apply Fix to '{col}'", key=fix_key):
                work = st.session_state.cleaned_df
                before_null = work[col].isnull().sum()
                try:
                    if strat == 'Drop rows':
                        work = work.dropna(subset=[col])

                    elif strat == 'Fill Mean':
                        mv = pd.to_numeric(work[col], errors='coerce').mean()
                        if pd.isna(mv):
                            wbox("Cannot compute mean — column has no numeric values. Use Mode or Custom.")
                        else:
                            work[col] = work[col].fillna(mv)

                    elif strat == 'Fill Median':
                        mv = pd.to_numeric(work[col], errors='coerce').median()
                        if pd.isna(mv):
                            wbox("Cannot compute median — column has no numeric values.")
                        else:
                            work[col] = work[col].fillna(mv)

                    elif strat == 'Fill Mode':
                        mode_vals = work[col].mode()
                        if len(mode_vals) == 0:
                            wbox("Cannot compute mode — column is entirely null.")
                        else:
                            work[col] = work[col].fillna(mode_vals[0])

                    elif strat == 'Fill Custom':
                        if cval is None or str(cval).strip() == '':
                            wbox("Please enter a custom value before applying.")
                        else:
                            work[col] = work[col].fillna(cval)

                    after_null = work[col].isnull().sum()
                    st.session_state.cleaned_df = work
                    sbox(f"Fixed <b>{before_null - after_null:,}</b> missing cells in <b>{col}</b>. {after_null} remain.")
                    st.rerun()
                except Exception as e:
                    dbox(f"Fix failed for {col}: {e}")

    divider()
    # Fix all at once option
    st.markdown("##### ⚡ Fix All Missing Values At Once")
    fa1,fa2 = st.columns(2)
    with fa1:
        all_strat = st.selectbox("Strategy for ALL columns", [
            'Fill Numeric → Mean, Text → Mode',
            'Fill Numeric → Median, Text → Mode',
            'Drop all rows with any null',
            'Fill everything with 0'
        ], key='all_strat')
    with fa2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("⚡ Apply to All Missing Columns", key='fix_all'):
            work = st.session_state.cleaned_df
            try:
                if all_strat == 'Drop all rows with any null':
                    work = work.dropna()

                elif all_strat == 'Fill everything with 0':
                    work = work.fillna(0)

                elif 'Mean' in all_strat:
                    for c in work.columns:
                        if work[c].isnull().sum() > 0:
                            if str(work[c].dtype) == 'object':
                                modes = work[c].mode()
                                if len(modes) > 0:
                                    work[c] = work[c].fillna(modes[0])
                            else:
                                mv = pd.to_numeric(work[c], errors='coerce').mean()
                                work[c] = work[c].fillna(mv if not pd.isna(mv) else 0)

                elif 'Median' in all_strat:
                    for c in work.columns:
                        if work[c].isnull().sum() > 0:
                            if str(work[c].dtype) == 'object':
                                modes = work[c].mode()
                                if len(modes) > 0:
                                    work[c] = work[c].fillna(modes[0])
                            else:
                                mv = pd.to_numeric(work[c], errors='coerce').median()
                                work[c] = work[c].fillna(mv if not pd.isna(mv) else 0)

                st.session_state.cleaned_df = work
                sbox("Applied bulk fix. Refreshing...")
                st.rerun()
            except Exception as e:
                dbox(f"Bulk fix failed: {e}")

st.markdown('</div>', unsafe_allow_html=True)


# ── 5C: DROP COLUMNS ─────────────────────────────────────────
st.markdown('<div class="iq-section">', unsafe_allow_html=True)
st.markdown("#### ❌ Drop Unwanted Columns")

df = st.session_state.cleaned_df

# Smart suggestions for which columns to consider dropping
suggest_drop = []
for c in df.columns:
    if df[c].nunique() == len(df):
        suggest_drop.append(f"{c} (all unique — likely ID)")
    elif df[c].nunique() == 1:
        suggest_drop.append(f"{c} (only 1 unique value — no information)")
    elif df[c].isnull().sum() / max(len(df),1) > 0.5:
        suggest_drop.append(f"{c} (>50% missing)")

if suggest_drop:
    wbox("Suggested columns to consider dropping:<br>" + "<br>".join([f"&nbsp;&nbsp;• {s}" for s in suggest_drop]))

cols_drop = st.multiselect("Select columns to remove", options=df.columns.tolist(), key='cols_drop')
if st.button("❌ Drop Selected Columns", key='do_drop'):
    if cols_drop:
        st.session_state.cleaned_df = st.session_state.cleaned_df.drop(columns=cols_drop)
        sbox(f"Dropped: {cols_drop}")
        st.rerun()
    else:
        wbox("Select at least one column first.")

st.markdown('</div>', unsafe_allow_html=True)


# ── 5D: FINAL CLEANED SUMMARY ────────────────────────────────
st.markdown('<div class="iq-section">', unsafe_allow_html=True)
st.markdown("#### ✅ Final Cleaned Dataset")

df = st.session_state.cleaned_df
qs, gr = quality_score(df)

fs1,fs2,fs3,fs4,fs5 = st.columns(5)
fs1.metric("Final Rows",      f"{df.shape[0]:,}", delta=f"{df.shape[0]-data.shape[0]:,}")
fs2.metric("Final Columns",   f"{df.shape[1]}",   delta=f"{df.shape[1]-data.shape[1]}")
fs3.metric("Missing Cells",   f"{df.isnull().sum().sum():,}")
fs4.metric("Duplicates",      f"{df.duplicated().sum():,}")
fs5.metric("Quality Score",   f"{qs}/100 {gr}")

st.dataframe(df, use_container_width=True)

dl1,dl2 = st.columns(2)
with dl1:
    st.download_button(
        "⬇️ Download Cleaned CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="cleaned_data.csv", mime="text/csv", key='dl_csv'
    )
with dl2:
    try:
        import io
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine='openpyxl')
        st.download_button(
            "⬇️ Download Cleaned Excel",
            data=buf.getvalue(),
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key='dl_xlsx'
        )
    except Exception:
        pass

st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# SECTION 6 — AUTO INSIGHTS
# ════════════════════════════════════════════════════════════════
divider()
section("📊", "Step 5 — Automated Data Intelligence")

adf      = st.session_state.cleaned_df
num_cols = adf.select_dtypes(include='number').columns.tolist()
cat_cols = adf.select_dtypes(include='object').columns.tolist()

# Guard against empty dataset in analysis
if len(adf) == 0:
    dbox("Cleaned dataset is empty. Reset and re-upload.")
    st.stop()

# KPI overview
o1,o2,o3,o4,o5 = st.columns(5)
o1.metric("Rows",         f"{adf.shape[0]:,}")
o2.metric("Columns",      f"{adf.shape[1]}")
o3.metric("Numeric",      f"{len(num_cols)}")
o4.metric("Categorical",  f"{len(cat_cols)}")
o5.metric("Memory",       f"{adf.memory_usage(deep=True).sum()/1024:.1f} KB")
st.markdown("<br>", unsafe_allow_html=True)

# Quality score panel
qs, gr = quality_score(adf)
qs_1,qs_2,qs_3 = st.columns(3)
qs_1.metric("Quality Score", f"{qs}/100")
qs_2.metric("Grade",         gr)
qs_3.metric("Missing %",     f"{adf.isnull().sum().sum()/max(adf.shape[0]*adf.shape[1],1)*100:.2f}%")
st.markdown("<br>", unsafe_allow_html=True)

# Numeric analysis
if num_cols:
    st.markdown("#### 🔢 Numeric Column Intelligence")
    for col in num_cols:
        with st.expander(f"📊  {col}"):
            cd = pd.to_numeric(adf[col], errors='coerce').dropna()

            if len(cd) == 0:
                wbox(f"{col} has no valid numeric values after coercion.")
                continue

            mean_v = float(cd.mean()); med_v = float(cd.median())
            std_v  = float(cd.std());  skw_v = float(cd.skew()) if len(cd)>=3 else 0.0
            mn_v   = float(cd.min());  mx_v  = float(cd.max())
            q1_v   = float(cd.quantile(.25)); q3_v = float(cd.quantile(.75))
            iqr_v  = q3_v - q1_v
            out_n  = int(((cd < q1_v-1.5*iqr_v)|(cd > q3_v+1.5*iqr_v)).sum())

            nc1,nc2,nc3,nc4,nc5,nc6 = st.columns(6)
            nc1.metric("Mean",    f"{mean_v:.3f}")
            nc2.metric("Median",  f"{med_v:.3f}")
            nc3.metric("Std Dev", f"{std_v:.3f}")
            nc4.metric("Min",     f"{mn_v:.3f}")
            nc5.metric("Max",     f"{mx_v:.3f}")
            nc6.metric("Outliers",f"{out_n}")

            ins = []
            diff = abs(mean_v-med_v)/(abs(med_v)+1e-9)*100
            ins.append(f"Mean vs Median gap: {diff:.1f}% — {'outliers are skewing the average.' if diff>10 else 'distribution is symmetric.'}")
            ins.append(f"Skewness {skw_v:.2f} → {'right-skewed, use Median for filling.' if skw_v>1 else 'left-skewed, use Median for filling.' if skw_v<-1 else 'normal, Mean is reliable.'}")
            cv = std_v/(abs(mean_v)+1e-9)*100
            ins.append(f"Coefficient of Variation: {cv:.1f}% — {'high spread, heterogeneous data.' if cv>50 else 'low spread, consistent values.'}")
            ins.append(f"{out_n} outlier(s) = {out_n/max(len(cd),1)*100:.1f}% — {'review before ML.' if out_n>0 else 'no outliers detected.'}")
            ins.append(f"Range: {mn_v:.3f} → {mx_v:.3f} (spread = {mx_v-mn_v:.3f})")

            st.markdown("**📝 Intelligence:**")
            for i in ins: st.markdown(f"- {i}")

# Categorical analysis
if cat_cols:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔤 Categorical Column Intelligence")
    for col in cat_cols:
        with st.expander(f"🏷️  {col}"):
            cd    = adf[col].dropna()
            if len(cd) == 0:
                wbox(f"{col} has no non-null values.")
                continue
            vc    = cd.value_counts()
            uniq  = int(cd.nunique())
            top_v = str(vc.index[0]); top_c = int(vc.iloc[0])
            bot_v = str(vc.index[-1]); bot_c = int(vc.iloc[-1])
            dom   = top_c/max(len(cd),1)*100

            cc1,cc2,cc3,cc4 = st.columns(4)
            cc1.metric("Unique",    f"{uniq:,}")
            cc2.metric("Top Value", top_v[:20])
            cc3.metric("Top Count", f"{top_c:,}")
            cc4.metric("Dominance", f"{dom:.1f}%")

            vc_df = vc.head(15).reset_index()
            vc_df.columns = ['Value','Count']
            vc_df['%'] = (vc_df['Count']/max(len(cd),1)*100).round(2)
            st.dataframe(vc_df, use_container_width=True)

            ins = []
            if uniq == 1:
                ins.append(f"Single value '{top_v}' — zero variance. Drop this column.")
            elif uniq >= len(cd):
                ins.append("All values unique — likely an ID. Drop before ML.")
            elif uniq <= 5:
                ins.append(f"Low cardinality ({uniq}) — ideal for one-hot encoding.")
            elif uniq > 50:
                ins.append(f"High cardinality ({uniq}) — use target/frequency encoding.")
            else:
                ins.append(f"Moderate cardinality ({uniq} values).")
            if dom > 80:
                ins.append(f"'{top_v}' dominates at {dom:.1f}% — class imbalance risk.")
            elif dom > 50:
                ins.append(f"'{top_v}' is majority at {dom:.1f}%.")
            else:
                ins.append(f"Balanced distribution. Top value holds {dom:.1f}%.")
            ins.append(f"Rarest value: '{bot_v}' with {bot_c} occurrence(s).")

            st.markdown("**📝 Intelligence:**")
            for i in ins: st.markdown(f"- {i}")

# Correlation table
if len(num_cols) >= 2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔗 Correlation Intelligence")
    try:
        corr = adf[num_cols].apply(pd.to_numeric, errors='coerce').corr()
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                v = corr.iloc[i,j]
                if not np.isnan(v):
                    pairs.append({
                        'Column A':corr.columns[i],
                        'Column B':corr.columns[j],
                        'Correlation':round(float(v),4),
                        'Strength': '🔴 Strong' if abs(v)>=.7 else '🟡 Moderate' if abs(v)>=.4 else '⚪ Weak',
                        'Direction':'↗ Positive' if v>0 else '↘ Negative'
                    })
        if pairs:
            pairs_df = pd.DataFrame(pairs).sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(pairs_df, use_container_width=True)
            strong = pairs_df[abs(pairs_df['Correlation'])>=.7]
            if not strong.empty:
                st.markdown("**Strong Correlations:**")
                for _,row in strong.iterrows():
                    ibox(f"<b>{row['Column A']}</b> ↔ <b>{row['Column B']}</b> = {row['Correlation']:.4f} {row['Direction']}")
            else:
                ibox("No strong correlations (|r| ≥ 0.7) found.")
        else:
            ibox("Not enough numeric data to compute correlations.")
    except Exception as e:
        wbox(f"Correlation error: {e}")

# Outlier summary
if num_cols:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🚨 Outlier Intelligence")
    out_rows = []
    for col in num_cols:
        try:
            cd   = pd.to_numeric(adf[col], errors='coerce').dropna()
            if len(cd) < 4: continue
            q1_  = float(cd.quantile(.25)); q3_ = float(cd.quantile(.75))
            iqr_ = q3_ - q1_
            oc   = int(((cd<q1_-1.5*iqr_)|(cd>q3_+1.5*iqr_)).sum())
            op   = oc/max(len(cd),1)*100
            out_rows.append({
                'Column'  :col,
                'Count'   :oc,
                'Outlier %':round(op,2),
                'Status'  :'🔴 Investigate' if op>5 else '🟡 Monitor' if op>0 else '🟢 Clean',
                'Lower Fence':round(q1_-1.5*iqr_,3),
                'Upper Fence':round(q3_+1.5*iqr_,3)
            })
        except Exception:
            continue
    if out_rows:
        st.dataframe(pd.DataFrame(out_rows).sort_values('Outlier %',ascending=False), use_container_width=True)

# Auto summary report
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### 📋 Auto-Generated Executive Summary")

lines = []
lines.append(f"Dataset: **{adf.shape[0]:,} rows × {adf.shape[1]} columns** after cleaning.")
lines.append(f"**{len(num_cols)} numeric** and **{len(cat_cols)} categorical** columns.")
lines.append("**No missing values.**" if adf.isnull().sum().sum()==0 else f"**{adf.isnull().sum().sum():,} missing cells** remain — further cleaning recommended.")
lines.append("**No duplicates.**" if adf.duplicated().sum()==0 else f"**{adf.duplicated().sum():,} duplicate rows** still present.")
qs2, gr2 = quality_score(adf)
lines.append(f"Data quality score: **{qs2}/100** — {gr2}.")

if len(num_cols)>=2 and 'pairs_df' in dir():
    try:
        sc = len(pairs_df[abs(pairs_df['Correlation'])>=.7])
        lines.append(f"**{sc} strong correlation(s)** detected." if sc>0 else "No strong correlations detected.")
    except Exception: pass

if out_rows:
    worst = max(out_rows, key=lambda x: x['Outlier %'])
    if worst['Outlier %']>5:
        lines.append(f"**{worst['Column']}** has the highest outlier rate at {worst['Outlier %']}% — review before modelling.")

if cat_cols:
    hc = [c for c in cat_cols if adf[c].nunique()>50]
    if hc:
        lines.append(f"High-cardinality columns requiring encoding: **{', '.join(hc)}**.")

for ln in lines:
    st.markdown(f"- {ln}")

st.download_button(
    "⬇️ Download Executive Summary",
    data='\n'.join([l.replace('**','') for l in lines]),
    file_name="executive_summary.txt",
    mime="text/plain",
    key='dl_summary'
)

divider()
st.markdown("<div style='text-align:center;color:#1f2937;font-size:.78em;padding:8px 0;'>DataIQ Analytics Suite · Built with Streamlit &amp; Pandas · Professional Data Intelligence</div>", unsafe_allow_html=True)