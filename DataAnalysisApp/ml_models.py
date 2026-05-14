from typing import Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import logging

logger = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score
)

# Classification models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Regression models
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                  SGDRegressor, HuberRegressor)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Clustering models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False


def ibox(t):
    st.markdown(f'<div style="background:#0c1230;border-left:3px solid #6366f1;padding:12px 16px;border-radius:0 10px 10px 0;color:#a5b4fc;font-size:.84em;line-height:1.9;margin:8px 0;">💡 {t}</div>', unsafe_allow_html=True)


def sbox(t):
    st.markdown(f'<div style="background:#061a0f;border-left:3px solid #22c55e;padding:12px 16px;border-radius:0 10px 10px 0;color:#86efac;font-size:.84em;line-height:1.9;margin:8px 0;">✅ {t}</div>', unsafe_allow_html=True)


def wbox(t):
    st.markdown(f'<div style="background:#1a140a;border-left:3px solid #f59e0b;padding:12px 16px;border-radius:0 10px 10px 0;color:#fcd34d;font-size:.84em;line-height:1.9;margin:8px 0;">⚠️ {t}</div>', unsafe_allow_html=True)


def dbox(t):
    st.markdown(f'<div style="background:#1a0808;border-left:3px solid #ef4444;padding:12px 16px;border-radius:0 10px 10px 0;color:#fca5a5;font-size:.84em;line-height:1.9;margin:8px 0;">🔴 {t}</div>', unsafe_allow_html=True)


def divider():
    st.markdown('<hr style="border:none;border-top:1px solid #1e293b;margin:16px 0;">', unsafe_allow_html=True)


def safe_float(v):
    try:
        f = float(v)
        return 0.0 if (np.isnan(f) or np.isinf(f)) else f
    except:
        return 0.0


def prepare_ml_data(df, features, target) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    try:
        ml_df = df[features + [target]].copy()
        ml_df = ml_df.dropna()
        if len(ml_df) < 10:
            return None, None, "Need at least 10 rows after dropping nulls."

        le = LabelEncoder()
        for col in ml_df.select_dtypes(include='object').columns:
            ml_df[col] = le.fit_transform(ml_df[col].astype(str))

        X = np.asarray(ml_df[features], dtype=np.float32)
        y = np.asarray(ml_df[target], dtype=object)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        logger.info(f"Prepared ML data: X shape {X.shape}, y shape {y.shape}")
        return X, y, None
    except Exception as e:
        logger.error(f"Error preparing ML data: {e}")
        return None, None, str(e)


def show_classification_metrics(y_test, y_pred, y_prob=None, model_name=""):
    st.markdown(f"#### 📊 {model_name} — Results")
    acc = safe_float(accuracy_score(y_test, y_pred))
    prec = safe_float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    rec = safe_float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1 = safe_float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc*100:.2f}%")
    c2.metric("Precision", f"{prec*100:.2f}%")
    c3.metric("Recall", f"{rec*100:.2f}%")
    c4.metric("F1 Score", f"{f1*100:.2f}%")

    logger.info(f"Classification metrics - Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}")


def show_regression_metrics(y_test, y_pred, model_name=""):
    st.markdown(f"#### 📊 {model_name} — Results")
    mse = safe_float(mean_squared_error(y_test, y_pred))
    rmse = safe_float(np.sqrt(mse))
    mae = safe_float(mean_absolute_error(y_test, y_pred))
    r2 = safe_float(r2_score(y_test, y_pred))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{r2:.4f}")
    c2.metric("RMSE", f"{rmse:.4f}")
    c3.metric("MAE", f"{mae:.4f}")
    c4.metric("MSE", f"{mse:.4f}")

    logger.info(f"Regression metrics - R²: {r2}, RMSE: {rmse}, MAE: {mae}")


def run_ml_models(df):
    """Main ML models interface"""
    logger.info("ML Models interface started")
    
    st.markdown("""
    <div style='background:linear-gradient(145deg,#0f172a,#111827);border:1px solid #1e293b;
        border-radius:16px;padding:22px 26px;margin:10px 0;'>
    <div style='font-size:1.1em;font-weight:800;color:#f1f5f9;margin-bottom:4px;'>
        🤖 Machine Learning Models
    </div>
    <div style='color:#475569;font-size:.82em;'>
        Supervised learning with classification, regression, and clustering
    </div>
    </div>
    """, unsafe_allow_html=True)

    if df is None or len(df) == 0:
        dbox("No data available. Complete preprocessing first.")
        logger.warning("No data available for ML models")
        return

    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()

    ml_tabs = st.tabs([
        "🎯 Classification",
        "📈 Regression",
        "🔵 Clustering",
        "🏆 Model Comparison"
    ])

    # CLASSIFICATION TAB
    with ml_tabs[0]:
        st.markdown("#### 🎯 Classification Models")
        ibox("Classification predicts a category (which class does this belong to?).")

        target_cls = st.selectbox("Target column (what to predict)", all_cols, key='cls_tgt')
        feature_cls = st.multiselect("Feature columns (inputs)",
                                      [c for c in all_cols if c != target_cls],
                                      default=[c for c in num_cols if c != target_cls][:5],
                                      key='cls_feat')

        if not feature_cls:
            wbox("Select at least one feature column.")
        else:
            cls_model = st.selectbox("Select classification model", [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "SVM",
                "KNN"
            ], key='cls_model')

            test_size = st.slider("Test size %", 10, 40, 20, key='cls_ts')
            cv_folds = st.slider("Cross-validation folds", 2, 10, 5, key='cls_cv')

            if st.button("🚀 Train Classification Model", key='train_cls'):
                logger.info(f"Training {cls_model} for classification")
                X, y, err = prepare_ml_data(df, feature_cls, target_cls)
                if err:
                    dbox(err)
                else:
                    assert X is not None and y is not None
                    try:
                        if y.dtype == object or str(y.dtype) == 'object':
                            le_y = LabelEncoder()
                            y = le_y.fit_transform(y.astype(str))

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size/100, random_state=42,
                            stratify=y if len(np.unique(y)) > 1 else None
                        )

                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)

                        with st.spinner(f"Training {cls_model}..."):
                            model_map = {
                                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                                "Decision Tree": DecisionTreeClassifier(random_state=42),
                                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                                "SVM": SVC(probability=True, random_state=42),
                                "KNN": KNeighborsClassifier(n_neighbors=5)
                            }

                            model = model_map[cls_model]
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            show_classification_metrics(y_test, y_pred, None, cls_model)

                            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                            cv_s = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

                            divider()
                            cv1, cv2 = st.columns(2)
                            cv1.metric(f"CV Mean Accuracy", f"{cv_s.mean()*100:.2f}%")
                            cv2.metric("CV Std Dev", f"±{cv_s.std()*100:.2f}%")

                            sbox(f"<b>{cls_model}</b> trained successfully!")
                            logger.info(f"{cls_model} trained. CV Accuracy: {cv_s.mean():.4f}")

                    except Exception as e:
                        dbox(f"Training failed: {e}")
                        logger.error(f"Error training {cls_model}: {e}")

    # REGRESSION TAB
    with ml_tabs[1]:
        st.markdown("#### 📈 Regression Models")
        ibox("Regression predicts a continuous number.")

        target_reg = st.selectbox("Target column (numeric to predict)", num_cols, key='reg_tgt')
        feature_reg = st.multiselect("Feature columns",
                                      [c for c in all_cols if c != target_reg],
                                      default=[c for c in num_cols if c != target_reg][:5],
                                      key='reg_feat')

        if not feature_reg:
            wbox("Select feature columns.")
        else:
            reg_model = st.selectbox("Select regression model", [
                "Linear Regression",
                "Ridge",
                "Lasso",
                "Random Forest",
                "Gradient Boosting"
            ], key='reg_model')

            reg_ts = st.slider("Test size %", 10, 40, 20, key='reg_ts')
            reg_cv = st.slider("CV folds", 2, 10, 5, key='reg_cv')

            if st.button("🚀 Train Regression Model", key='train_reg'):
                logger.info(f"Training {reg_model} for regression")
                X, y, err = prepare_ml_data(df, feature_reg, target_reg)
                if err:
                    dbox(err)
                else:
                    assert X is not None and y is not None
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=reg_ts/100, random_state=42
                        )
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)

                        with st.spinner(f"Training {reg_model}..."):
                            model_map = {
                                "Linear Regression": LinearRegression(),
                                "Ridge": Ridge(),
                                "Lasso": Lasso(),
                                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                                "Gradient Boosting": GradientBoostingRegressor(random_state=42)
                            }

                            model = model_map[reg_model]
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            show_regression_metrics(y_test, y_pred, reg_model)

                            kf = KFold(n_splits=reg_cv, shuffle=True, random_state=42)
                            cv_s = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)

                            divider()
                            cv1, cv2 = st.columns(2)
                            cv1.metric("CV Mean R²", f"{cv_s.mean():.4f}")
                            cv2.metric("CV Std Dev", f"±{cv_s.std():.4f}")

                            sbox(f"<b>{reg_model}</b> trained successfully!")
                            logger.info(f"{reg_model} trained. CV R²: {cv_s.mean():.4f}")

                    except Exception as e:
                        dbox(f"Training failed: {e}")
                        logger.error(f"Error training {reg_model}: {e}")

    # CLUSTERING TAB
    with ml_tabs[2]:
        st.markdown("#### 🔵 Clustering Models")
        ibox("Clustering finds natural groups in your data.")

        clust_feat = st.multiselect("Feature columns for clustering",
                                     num_cols, default=num_cols[:3], key='clust_feat')

        if clust_feat:
            n_clusters = st.slider("Number of clusters", 2, 10, 3, key='n_clust')

            if st.button("🚀 Run Clustering", key='run_clust'):
                logger.info(f"Running clustering with {n_clusters} clusters")
                X_cl = df[clust_feat].copy()
                X_cl = X_cl.apply(pd.to_numeric, errors='coerce').fillna(0).values
                X_cl = StandardScaler().fit_transform(X_cl)

                with st.spinner("Running clustering..."):
                    try:
                        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        labels = model.fit_predict(X_cl)

                        st.metric("Clusters", n_clusters)
                        sbox("Clustering complete!")
                        logger.info(f"Clustering completed with {n_clusters} clusters")

                    except Exception as e:
                        dbox(f"Clustering failed: {e}")
                        logger.error(f"Clustering error: {e}")

    # MODEL COMPARISON TAB
    with ml_tabs[3]:
        st.markdown("#### 🏆 Model Comparison")
        ibox("Compare multiple models automatically.")
        
        bench_type = st.selectbox("Task type", ["Classification", "Regression"], key='bench_type')
        st.info(f"Model comparison for {bench_type} coming soon...")
        logger.info("Model comparison interface accessed")
