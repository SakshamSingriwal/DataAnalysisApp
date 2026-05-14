import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import (
    LabelEncoder, OrdinalEncoder, OneHotEncoder,
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, Normalizer, PowerTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')


def ibox(t): st.markdown(f'<div style="background:#0c1230;border-left:3px solid #6366f1;padding:12px 16px;border-radius:0 10px 10px 0;color:#a5b4fc;font-size:.84em;line-height:1.9;margin:8px 0;">💡 {t}</div>', unsafe_allow_html=True)
def sbox(t): st.markdown(f'<div style="background:#061a0f;border-left:3px solid #22c55e;padding:12px 16px;border-radius:0 10px 10px 0;color:#86efac;font-size:.84em;line-height:1.9;margin:8px 0;">✅ {t}</div>', unsafe_allow_html=True)
def wbox(t): st.markdown(f'<div style="background:#1a140a;border-left:3px solid #f59e0b;padding:12px 16px;border-radius:0 10px 10px 0;color:#fcd34d;font-size:.84em;line-height:1.9;margin:8px 0;">⚠️ {t}</div>', unsafe_allow_html=True)
def dbox(t): st.markdown(f'<div style="background:#1a0808;border-left:3px solid #ef4444;padding:12px 16px;border-radius:0 10px 10px 0;color:#fca5a5;font-size:.84em;line-height:1.9;margin:8px 0;">🔴 {t}</div>', unsafe_allow_html=True)
def divider(): st.markdown('<hr style="border:none;border-top:1px solid #1e293b;margin:16px 0;">', unsafe_allow_html=True)

def run_preprocessing(df):
    # Full preprocessing pipeline UI
    st.markdown("""
    <div style='background:linear-gradient(145deg,#0f172a,#111827);border:1px solid #1e293b;
        border-radius:16px;padding:22px 26px;margin:10px 0;'>
    <div style='font-size:1.1em;font-weight:800;color:#f1f5f9;margin-bottom:4px;'>
        ⚙️ Preprocessing Pipeline
    </div>
    <div style='color:#475569;font-size:.82em;'>
        Encoding, scaling, imputation and feature engineering — all in one place
    </div>
    </div>
    """, unsafe_allow_html=True)

    if 'preprocessed_df' not in st.session_state:
        st.session_state.preprocessed_df = df.copy()
        st.session_state.encoders  = {}
        st.session_state.scalers   = {}
        st.session_state.prep_log  = []

    work_df  = st.session_state.preprocessed_df
    num_cols = work_df.select_dtypes(include='number').columns.tolist()
    cat_cols = work_df.select_dtypes(include='object').columns.tolist()

    # Current state metrics
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows",            f"{work_df.shape[0]:,}")
    c2.metric("Columns",         f"{work_df.shape[1]}")
    c3.metric("Numeric Cols",    f"{len(num_cols)}")
    c4.metric("Categorical Cols",f"{len(cat_cols)}")

    divider()

    prep_tabs = st.tabs([
        "🔤 Encoding",
        "📏 Scaling",
        "🩹 Imputation",
        "🔧 Feature Engineering",
        "📋 Pipeline Log",
        "⬇️ Export"
    ])

    # ════════════════════════════════════════════════
    # ENCODING TAB
    # ════════════════════════════════════════════════
    with prep_tabs[0]:
        st.markdown("#### 🔤 Categorical Encoding")
        ibox("""
        ML models only understand numbers. Encoding converts text categories into numbers.<br><br>
        <b>Label Encoding:</b> Assigns a number to each category (Male=0, Female=1). Good for ordinal data.<br>
        <b>One-Hot Encoding:</b> Creates a new binary column for each category. Best for nominal data.<br>
        <b>Ordinal Encoding:</b> You define the order manually (Small=0, Medium=1, Large=2).<br>
        <b>Frequency Encoding:</b> Replaces category with how often it appears. Good for high cardinality.<br>
        <b>Target Encoding:</b> Replaces category with mean of target variable. Powerful but can overfit.
        """)

        if not cat_cols:
            sbox("No categorical columns found — encoding not needed.")
        else:
            enc_col    = st.selectbox("Select column to encode", cat_cols, key='enc_col')
            enc_method = st.selectbox("Encoding method", [
                "Label Encoding",
                "One-Hot Encoding",
                "Frequency Encoding",
                "Ordinal Encoding (manual order)",
                "Binary Encoding (0/1 for 2 categories)"
            ], key='enc_method')

            col_uniq = work_df[enc_col].dropna().unique()

            if enc_method == "One-Hot Encoding":
                wbox(f"This will add {len(col_uniq)} new columns. High cardinality (>{30} categories) will bloat the dataset.") if len(col_uniq)>10 else None

            if enc_method == "Ordinal Encoding (manual order)":
                st.markdown("**Drag or type the order of categories (first = lowest rank):**")
                ordinal_order = st.text_area(
                    "Enter categories in order (one per line)",
                    value="\n".join([str(v) for v in col_uniq[:20]]),
                    key='ord_order'
                )

            target_enc_col = None
            if "Target" in enc_method:
                target_enc_col = st.selectbox("Target column (numeric)", num_cols, key='tgt_enc')

            if st.button(f"Apply {enc_method}", key='apply_enc'):
                work = st.session_state.preprocessed_df.copy()
                try:
                    if enc_method == "Label Encoding":
                        le = LabelEncoder()
                        work[enc_col] = le.fit_transform(work[enc_col].astype(str))
                        st.session_state.encoders[enc_col] = le
                        msg = f"Label encoded '{enc_col}' — {len(le.classes_)} classes"

                    elif enc_method == "One-Hot Encoding":
                        dummies = pd.get_dummies(work[enc_col], prefix=enc_col, drop_first=False)
                        work    = pd.concat([work.drop(columns=[enc_col]), dummies], axis=1)
                        msg     = f"One-hot encoded '{enc_col}' — added {len(dummies.columns)} columns"

                    elif enc_method == "Frequency Encoding":
                        freq_map = work[enc_col].value_counts(normalize=True).to_dict()
                        work[enc_col+'_freq'] = work[enc_col].map(freq_map)
                        work = work.drop(columns=[enc_col])
                        msg  = f"Frequency encoded '{enc_col}'"

                    elif enc_method == "Ordinal Encoding (manual order)":
                        order    = [o.strip() for o in ordinal_order.split('\n') if o.strip()]
                        ord_map  = {v:i for i,v in enumerate(order)}
                        work[enc_col] = work[enc_col].astype(str).map(ord_map)
                        msg = f"Ordinal encoded '{enc_col}' with custom order"

                    elif enc_method == "Binary Encoding (0/1 for 2 categories)":
                        if len(col_uniq) != 2:
                            wbox(f"Binary encoding requires exactly 2 categories. '{enc_col}' has {len(col_uniq)}.")
                            st.stop()
                        work[enc_col] = (work[enc_col] == col_uniq[0]).astype(int)
                        msg = f"Binary encoded '{enc_col}' ('{col_uniq[0]}'=1, '{col_uniq[1]}'=0)"

                    st.session_state.preprocessed_df = work
                    st.session_state.prep_log.append(f"[Encoding] {msg}")
                    sbox(msg)
                    st.dataframe(work.head(5), use_container_width=True)
                    st.rerun()
                except Exception as e:
                    dbox(f"Encoding failed: {e}")

    # ════════════════════════════════════════════════
    # SCALING TAB
    # ════════════════════════════════════════════════
    with prep_tabs[1]:
        st.markdown("#### 📏 Feature Scaling")
        ibox("""
        Scaling ensures all numeric features are on the same range.
        Without scaling, large-magnitude features dominate the model.<br><br>
        <b>Standard Scaler (Z-score):</b> Mean=0, Std=1. Best for normally distributed data.<br>
        <b>Min-Max Scaler:</b> Scales to [0,1]. Good for neural networks and KNN.<br>
        <b>Robust Scaler:</b> Uses median and IQR — not affected by outliers.<br>
        <b>Max-Abs Scaler:</b> Scales to [-1,1]. Good for sparse data.<br>
        <b>Power Transform (Yeo-Johnson):</b> Makes data more Gaussian — great for skewed data.<br>
        <b>Normalizer:</b> Scales each ROW to unit norm — used for text/NLP.
        """)

        work_df  = st.session_state.preprocessed_df
        num_cols2 = work_df.select_dtypes(include='number').columns.tolist()

        if not num_cols2:
            wbox("No numeric columns available for scaling.")
        else:
            sc_cols   = st.multiselect("Select columns to scale",
                                        num_cols2, default=num_cols2, key='sc_cols')
            sc_method = st.selectbox("Scaling method", [
                "Standard Scaler (Z-score normalization)",
                "Min-Max Scaler [0, 1]",
                "Robust Scaler (outlier resistant)",
                "Max-Abs Scaler [-1, 1]",
                "Power Transform (Yeo-Johnson)",
                "Normalizer (row-wise)"
            ], key='sc_method')

            # Show before stats
            if sc_cols:
                st.markdown("**Before Scaling — Statistics:**")
                try:
                    st.dataframe(
                        work_df[sc_cols].describe().T.style.background_gradient(cmap='Blues').format(precision=3),
                        use_container_width=True
                    )
                except:
                    st.dataframe(work_df[sc_cols].describe().T, use_container_width=True)

            if st.button(f"Apply Scaling", key='apply_scale') and sc_cols:
                work = st.session_state.preprocessed_df.copy()
                try:
                    scaler_map = {
                        "Standard Scaler (Z-score normalization)": StandardScaler(),
                        "Min-Max Scaler [0, 1]":                   MinMaxScaler(),
                        "Robust Scaler (outlier resistant)":        RobustScaler(),
                        "Max-Abs Scaler [-1, 1]":                   MaxAbsScaler(),
                        "Power Transform (Yeo-Johnson)":            PowerTransformer(method='yeo-johnson'),
                        "Normalizer (row-wise)":                    Normalizer()
                    }
                    scaler = scaler_map[sc_method]
                    work[sc_cols] = scaler.fit_transform(work[sc_cols].fillna(0))
                    st.session_state.scalers[sc_method] = scaler
                    st.session_state.preprocessed_df    = work
                    msg = f"Applied {sc_method} to {len(sc_cols)} columns"
                    st.session_state.prep_log.append(f"[Scaling] {msg}")
                    sbox(msg)

                    st.markdown("**After Scaling — Statistics:**")
                    try:
                        st.dataframe(
                            work[sc_cols].describe().T.style.background_gradient(cmap='Greens').format(precision=3),
                            use_container_width=True
                        )
                    except:
                        st.dataframe(work[sc_cols].describe().T, use_container_width=True)
                    st.rerun()
                except Exception as e:
                    dbox(f"Scaling failed: {e}")

    # ════════════════════════════════════════════════
    # IMPUTATION TAB
    # ════════════════════════════════════════════════
    with prep_tabs[2]:
        st.markdown("#### 🩹 Advanced Imputation")
        ibox("""
        <b>Simple Imputer:</b> Fills with mean, median, most_frequent or constant.<br>
        <b>KNN Imputer:</b> Fills using K nearest neighbours — more accurate for complex patterns.<br>
        <b>Iterative Imputer:</b> Models each feature with missing values as a function of other features.
        """)
        work_df3 = st.session_state.preprocessed_df
        missing3 = work_df3.isnull().sum()
        miss_cols3 = missing3[missing3>0].index.tolist()

        if not miss_cols3:
            sbox("No missing values — imputation not needed.")
        else:
            imp_method = st.selectbox("Imputation strategy", [
                "Simple — Mean",
                "Simple — Median",
                "Simple — Most Frequent",
                "Simple — Constant",
                "KNN Imputer (k=5)"
            ], key='imp_method')

            const_val = ""
            if "Constant" in imp_method:
                const_val = st.text_input("Fill constant value", "0", key='const_v')

            imp_cols = st.multiselect("Columns to impute",
                                       miss_cols3, default=miss_cols3, key='imp_cols')

            if st.button("Apply Imputation", key='apply_imp') and imp_cols:
                work = st.session_state.preprocessed_df.copy()
                try:
                    if "KNN" in imp_method:
                        knn_imp = KNNImputer(n_neighbors=5)
                        num_imp_cols = [c for c in imp_cols if str(work[c].dtype)!='object']
                        if num_imp_cols:
                            work[num_imp_cols] = knn_imp.fit_transform(work[num_imp_cols])
                            msg = f"KNN imputed {len(num_imp_cols)} numeric columns"
                    else:
                        strat_map = {
                            "Simple — Mean":           "mean",
                            "Simple — Median":         "median",
                            "Simple — Most Frequent":  "most_frequent",
                            "Simple — Constant":       "constant"
                        }
                        strat = strat_map.get(imp_method, "mean")
                        for col in imp_cols:
                            if str(work[col].dtype) == 'object':
                                imp = SimpleImputer(strategy='most_frequent')
                            else:
                                fill_val = float(const_val) if strat=='constant' else None
                                imp = SimpleImputer(strategy=strat, fill_value=fill_val)
                            work[[col]] = imp.fit_transform(work[[col]])
                        msg = f"Imputed {len(imp_cols)} columns using {imp_method}"

                    st.session_state.preprocessed_df = work
                    st.session_state.prep_log.append(f"[Imputation] {msg}")
                    sbox(msg)
                    st.rerun()
                except Exception as e:
                    dbox(f"Imputation failed: {e}")

    # ════════════════════════════════════════════════
    # FEATURE ENGINEERING TAB
    # ════════════════════════════════════════════════
    with prep_tabs[3]:
        st.markdown("#### 🔧 Feature Engineering")
        ibox("Create new features from existing ones to improve model performance.")

        work_df4 = st.session_state.preprocessed_df
        num_cols4 = work_df4.select_dtypes(include='number').columns.tolist()

        fe_type = st.selectbox("Feature operation", [
            "Polynomial Features (x²)",
            "Log Transform (log1p)",
            "Square Root Transform",
            "Interaction Feature (col1 × col2)",
            "Ratio Feature (col1 / col2)",
            "Binning (cut into groups)",
            "Custom Formula"
        ], key='fe_type')

        if fe_type == "Polynomial Features (x²)":
            poly_col = st.selectbox("Column", num_cols4, key='poly_col')
            if st.button("Create x² feature", key='poly_btn'):
                work = st.session_state.preprocessed_df.copy()
                work[f"{poly_col}_squared"] = work[poly_col] ** 2
                st.session_state.preprocessed_df = work
                st.session_state.prep_log.append(f"[FE] Created {poly_col}_squared")
                sbox(f"Created '{poly_col}_squared'")
                st.rerun()

        elif fe_type == "Log Transform (log1p)":
            log_col = st.selectbox("Column", num_cols4, key='log_col')
            ibox("log1p(x) = log(x+1) — handles zero values safely. Use for right-skewed data.")
            if st.button("Apply Log Transform", key='log_btn'):
                work = st.session_state.preprocessed_df.copy()
                if work[log_col].min() < 0:
                    wbox("Column has negative values — log transform not applicable.")
                else:
                    work[f"{log_col}_log"] = np.log1p(work[log_col])
                    st.session_state.preprocessed_df = work
                    st.session_state.prep_log.append(f"[FE] Log transform on {log_col}")
                    sbox(f"Created '{log_col}_log'")
                    st.rerun()

        elif fe_type == "Square Root Transform":
            sqrt_col = st.selectbox("Column", num_cols4, key='sqrt_col')
            if st.button("Apply Sqrt", key='sqrt_btn'):
                work = st.session_state.preprocessed_df.copy()
                if work[sqrt_col].min() < 0:
                    wbox("Column has negative values — sqrt not applicable.")
                else:
                    work[f"{sqrt_col}_sqrt"] = np.sqrt(work[sqrt_col])
                    st.session_state.preprocessed_df = work
                    st.session_state.prep_log.append(f"[FE] Sqrt transform on {sqrt_col}")
                    sbox(f"Created '{sqrt_col}_sqrt'")
                    st.rerun()

        elif fe_type == "Interaction Feature (col1 × col2)":
            ia1 = st.selectbox("Column 1", num_cols4, key='ia1')
            ia2 = st.selectbox("Column 2", [c for c in num_cols4 if c!=ia1], key='ia2')
            if st.button("Create interaction", key='ia_btn'):
                work = st.session_state.preprocessed_df.copy()
                work[f"{ia1}_x_{ia2}"] = work[ia1] * work[ia2]
                st.session_state.preprocessed_df = work
                st.session_state.prep_log.append(f"[FE] Interaction {ia1} × {ia2}")
                sbox(f"Created '{ia1}_x_{ia2}'")
                st.rerun()

        elif fe_type == "Ratio Feature (col1 / col2)":
            r1c = st.selectbox("Numerator",   num_cols4, key='r1c')
            r2c = st.selectbox("Denominator", [c for c in num_cols4 if c!=r1c], key='r2c')
            if st.button("Create ratio", key='rat_btn'):
                work = st.session_state.preprocessed_df.copy()
                work[f"{r1c}_div_{r2c}"] = work[r1c] / (work[r2c].replace(0, np.nan) + 1e-9)
                st.session_state.preprocessed_df = work
                st.session_state.prep_log.append(f"[FE] Ratio {r1c}/{r2c}")
                sbox(f"Created '{r1c}_div_{r2c}'")
                st.rerun()

        elif fe_type == "Binning (cut into groups)":
            bin_col = st.selectbox("Column to bin", num_cols4, key='bin_col')
            n_bins  = st.slider("Number of bins", 2, 20, 5, key='n_bins')
            bin_method = st.radio("Method", ['Equal Width (cut)','Equal Frequency (qcut)'], key='bin_meth')
            if st.button("Apply Binning", key='bin_btn'):
                work = st.session_state.preprocessed_df.copy()
                try:
                    if 'Width' in bin_method:
                        work[f"{bin_col}_bin"] = pd.cut(work[bin_col], bins=n_bins, labels=False)
                    else:
                        work[f"{bin_col}_bin"] = pd.qcut(work[bin_col], q=n_bins, labels=False, duplicates='drop')
                    st.session_state.preprocessed_df = work
                    st.session_state.prep_log.append(f"[FE] Binned {bin_col} into {n_bins} bins")
                    sbox(f"Created '{bin_col}_bin'")
                    st.rerun()
                except Exception as e:
                    dbox(f"Binning failed: {e}")

        elif fe_type == "Custom Formula":
            st.markdown("Use `df` to reference the DataFrame. Example: `df['col1'] + df['col2']`")
            new_col_name = st.text_input("New column name", key='cust_name')
            formula      = st.text_area("Formula", key='cust_form', placeholder="df['col1'] * 2 + df['col2']")
            if st.button("Apply Formula", key='cust_btn'):
                if new_col_name and formula:
                    try:
                        work = st.session_state.preprocessed_df.copy()
                        df_ref = work.copy()
                        work[new_col_name] = eval(formula.replace('df', 'df_ref'))
                        st.session_state.preprocessed_df = work
                        st.session_state.prep_log.append(f"[FE] Custom: {new_col_name} = {formula}")
                        sbox(f"Created '{new_col_name}'")
                        st.rerun()
                    except Exception as e:
                        dbox(f"Formula error: {e}")

    # ════════════════════════════════════════════════
    # PIPELINE LOG TAB
    # ════════════════════════════════════════════════
    with prep_tabs[4]:
        st.markdown("#### 📋 Preprocessing Pipeline Log")
        log = st.session_state.get('prep_log', [])
        if not log:
            ibox("No preprocessing steps applied yet.")
        else:
            for i, step in enumerate(log, 1):
                st.markdown(f"""
                <div style='background:#0f172a;border:1px solid #1e293b;border-radius:8px;
                    padding:10px 16px;margin:4px 0;font-size:.82em;color:#e2e8f0;'>
                    <span style='color:#6366f1;font-weight:700;'>Step {i}</span>
                    &nbsp; {step}
                </div>
                """, unsafe_allow_html=True)

            if st.button("🗑️ Clear Log", key='clear_log'):
                st.session_state.prep_log = []
                st.rerun()

        divider()
        st.markdown("#### Current Preprocessed Dataset")
        work_final = st.session_state.preprocessed_df
        p1,p2,p3 = st.columns(3)
        p1.metric("Rows",    f"{work_final.shape[0]:,}")
        p2.metric("Columns", f"{work_final.shape[1]}")
        p3.metric("Missing", f"{work_final.isnull().sum().sum():,}")
        st.dataframe(work_final.head(10), use_container_width=True)

    # ════════════════════════════════════════════════
    # EXPORT TAB
    # ════════════════════════════════════════════════
    with prep_tabs[5]:
        st.markdown("#### ⬇️ Export Preprocessed Data")
        final_df = st.session_state.preprocessed_df
        sbox(f"Dataset ready — {final_df.shape[0]:,} rows × {final_df.shape[1]} columns")

        e1,e2,e3 = st.columns(3)
        with e1:
            st.download_button("⬇️ Download CSV",
                data=final_df.to_csv(index=False).encode('utf-8'),
                file_name="preprocessed_data.csv", mime="text/csv", key='dl_prep_csv')
        with e2:
            try:
                import io
                buf = io.BytesIO()
                final_df.to_excel(buf, index=False, engine='openpyxl')
                st.download_button("⬇️ Download Excel",
                    data=buf.getvalue(),
                    file_name="preprocessed_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key='dl_prep_xlsx')
            except: pass
        with e3:
            st.download_button("⬇️ Download JSON",
                data=final_df.to_json(orient='records', indent=2).encode('utf-8'),
                file_name="preprocessed_data.json", mime="application/json", key='dl_prep_json')

    return st.session_state.preprocessed_df