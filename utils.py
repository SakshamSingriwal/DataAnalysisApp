"""
DataIQ Pro Analytics - Utility Functions and UI Components

This module contains all helper functions and UI components for the DataIQ Pro Analytics application.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List


def ibox(text: str) -> None:
    """
    Renders a blue left-border info box with the text.
    
    Args:
        text: The text to display in the info box.
    """
    st.markdown(f"""
    <div style="
        background-color: #0f172a;
        border-left: 4px solid #6366f1;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: #e2e8f0;
    ">
        ℹ️ {text}
    </div>
    """, unsafe_allow_html=True)


def sbox(text: str) -> None:
    """
    Renders a green left-border success box.
    
    Args:
        text: The text to display in the success box.
    """
    st.markdown(f"""
    <div style="
        background-color: #0f172a;
        border-left: 4px solid #10b981;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: #e2e8f0;
    ">
        ✅ {text}
    </div>
    """, unsafe_allow_html=True)


def wbox(text: str) -> None:
    """
    Renders an orange left-border warning box.
    
    Args:
        text: The text to display in the warning box.
    """
    st.markdown(f"""
    <div style="
        background-color: #0f172a;
        border-left: 4px solid #f59e0b;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: #e2e8f0;
    ">
        ⚠️ {text}
    </div>
    """, unsafe_allow_html=True)


def dbox(text: str) -> None:
    """
    Renders a red left-border danger box.
    
    Args:
        text: The text to display in the danger box.
    """
    st.markdown(f"""
    <div style="
        background-color: #0f172a;
        border-left: 4px solid #ef4444;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: #e2e8f0;
    ">
        ❌ {text}
    </div>
    """, unsafe_allow_html=True)


def divider() -> None:
    """
    Renders a dark horizontal rule.
    """
    st.markdown("""
    <hr style="
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, #1e293b, #6366f1, #1e293b);
        margin: 20px 0;
    ">
    """, unsafe_allow_html=True)


def section(icon: str, title: str, sub: str, step_num: int) -> None:
    """
    Renders a dark card with left indigo border showing step number, icon, title and subtitle.
    
    Args:
        icon: The icon emoji or symbol.
        title: The section title.
        sub: The subtitle.
        step_num: The step number.
    """
    st.markdown(f"""
    <div style="
        background-color: #0f172a;
        border-left: 4px solid #6366f1;
        padding: 15px 20px;
        margin: 15px 0;
        border-radius: 8px;
        border: 1px solid #1e293b;
    ">
        <div style="
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        ">
            <div style="
                background: linear-gradient(135deg, #6366f1, #8b5cf6);
                color: white;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                margin-right: 12px;
                font-size: 14px;
            ">
                {step_num}
            </div>
            <span style="font-size: 24px; margin-right: 12px;">{icon}</span>
            <h3 style="
                margin: 0;
                background: linear-gradient(135deg, #6366f1, #8b5cf6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 20px;
            ">
                {title}
            </h3>
        </div>
        <p style="margin: 0; color: #94a3b8; font-size: 14px;">{sub}</p>
    </div>
    """, unsafe_allow_html=True)


def safe_float(value: Any) -> float:
    """
    Converts any value to float safely. Returns 0.0 if NaN, Inf or any exception.
    
    Args:
        value: The value to convert.
    
    Returns:
        The float value or 0.0 on failure.
    """
    try:
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return 0.0
        return result
    except:
        return 0.0


def fmt_bytes(bytes_val: int) -> str:
    """
    Converts bytes to human readable string (B, KB, MB, GB).
    
    Args:
        bytes_val: The number of bytes.
    
    Returns:
        Human readable string.
    """
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val/1024:.1f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val/1024**2:.1f} MB"
    else:
        return f"{bytes_val/1024**3:.1f} GB"


def compute_quality(df: pd.DataFrame) -> Tuple[float, str, float, float]:
    """
    Returns (score 0-100, grade string, missing_pct, dup_pct).
    Penalises missing values up to 40 points, duplicates up to 30 points.
    Grade: Excellent>=90, Good>=70, Fair>=50, Poor<50.
    Safe against empty or None DataFrames.
    
    Args:
        df: The DataFrame to analyze.
    
    Returns:
        Tuple of (score, grade, missing_pct, dup_pct).
    """
    if df is None or len(df) == 0:
        return 0.0, "Poor", 100.0, 0.0
    
    try:
        total_cells = len(df) * len(df.columns)
        if total_cells == 0:
            return 0.0, "Poor", 100.0, 0.0
        
        missing_cells = df.isnull().sum().sum()
        missing_pct = safe_float((missing_cells / total_cells) * 100)
        
        dup_rows = df.duplicated().sum()
        dup_pct = safe_float((dup_rows / len(df)) * 100) if len(df) > 0 else 0.0
        
        # Penalize missing values up to 40 points
        missing_penalty = min(missing_pct * 0.4, 40.0)
        
        # Penalize duplicates up to 30 points
        dup_penalty = min(dup_pct * 3.0, 30.0)  # 10% dupes = 30 points penalty
        
        score = 100.0 - missing_penalty - dup_penalty
        
        if score >= 90:
            grade = "Excellent"
        elif score >= 70:
            grade = "Good"
        elif score >= 50:
            grade = "Fair"
        else:
            grade = "Poor"
            
        return safe_float(score), grade, missing_pct, dup_pct
        
    except Exception as e:
        return 0.0, "Poor", 100.0, 0.0


def get_missing_recommendation(col: str, df: pd.DataFrame) -> Tuple[str, str]:
    """
    Returns (recommendation_string, severity_string).
    If missing < 5%: recommend drop rows.
    If text column: recommend mode with the actual mode value shown.
    If numeric and skewed (abs skew > 1): recommend median with value.
    If numeric and symmetric: recommend mean with value.
    All calculations wrapped in try/except.
    
    Args:
        col: The column name.
        df: The DataFrame.
    
    Returns:
        Tuple of (recommendation, severity).
    """
    try:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        
        if missing_pct < 5:
            return "Drop rows with missing values", "Low"
        
        if df[col].dtype == 'object':
            mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
            return f"Fill with mode: '{mode_val}'", "Medium"
        
        # Numeric column
        numeric_series = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(numeric_series) < 3:
            return "Drop rows with missing values", "Low"
        
        skew_val = numeric_series.skew()
        if abs(skew_val) > 1:
            median_val = numeric_series.median()
            return f"Fill with median: {median_val:.2f}", "Medium"
        else:
            mean_val = numeric_series.mean()
            return f"Fill with mean: {mean_val:.2f}", "Medium"
            
    except Exception as e:
        return "Drop rows with missing values", "Low"


def safe_numeric_stats(series: pd.Series) -> Optional[Dict[str, float]]:
    """
    Returns dict with: mean, median, std, min, max, skew, q1, q3, iqr, outliers, count, cv.
    Uses IQR method for outlier detection.
    Returns None if no valid numeric data.
    All values passed through safe_float.
    
    Args:
        series: The numeric series to analyze.
    
    Returns:
        Dict of statistics or None.
    """
    try:
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_series) < 1:
            return None
        
        q1 = numeric_series.quantile(0.25)
        q3 = numeric_series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = ((numeric_series < lower_bound) | (numeric_series > upper_bound)).sum()
        
        mean_val = numeric_series.mean()
        std_val = numeric_series.std()
        cv = std_val / mean_val if mean_val != 0 else 0
        
        return {
            'mean': safe_float(mean_val),
            'median': safe_float(numeric_series.median()),
            'std': safe_float(std_val),
            'min': safe_float(numeric_series.min()),
            'max': safe_float(numeric_series.max()),
            'skew': safe_float(numeric_series.skew() if len(numeric_series) >= 3 else 0),
            'q1': safe_float(q1),
            'q3': safe_float(q3),
            'iqr': safe_float(iqr),
            'outliers': safe_float(outliers),
            'count': safe_float(len(numeric_series)),
            'cv': safe_float(cv)
        }
        
    except Exception as e:
        return None


def prepare_features(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], str]:
    """
    Encodes object columns with LabelEncoder, removes NaN with dropna.
    Returns (X_array, y_array, encoders_dict, error_string).
    Replaces inf/nan in X with 0.
    
    Args:
        df: The DataFrame.
        feature_cols: List of feature column names.
        target_col: Target column name.
    
    Returns:
        Tuple of (X, y, encoders, error).
    """
    try:
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare data
        work_df = df[feature_cols + [target_col]].dropna()
        if len(work_df) == 0:
            return np.array([]), np.array([]), {}, "No valid data after dropping NaN"
        
        X = work_df[feature_cols].copy()
        y = work_df[target_col].copy()
        
        encoders = {}
        
        # Encode categorical features
        for col in feature_cols:
            if X[col].dtype == 'object':
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].astype(str))
                encoders[col] = encoder
        
        # Encode target if categorical
        if y.dtype == 'object':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
            encoders['target'] = target_encoder
        
        # Convert to numpy and handle inf/nan
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
        
        return X, y, encoders, ""
        
    except Exception as e:
        return np.array([]), np.array([]), {}, str(e)


def pipeline_banner(steps_done_list: List[bool]) -> None:
    """
    Shows horizontal pill-shaped steps.
    Done steps: green background, checkmark prefix.
    Current step: indigo background.
    Not done: dark background grey text.
    Arrow separators between steps.
    
    Args:
        steps_done_list: List of booleans indicating completion status.
    """
    step_names = [
        "Upload", "Explore", "Value Counts", "GroupBy", "Health Check", 
        "Analysis", "Hypothesis Testing", "Preprocessing", "ML Models", "Deep Learning", "Export"
    ]
    
    html_parts = []
    
    for i, (done, name) in enumerate(zip(steps_done_list, step_names)):
        if done:
            style = "background: linear-gradient(135deg, #10b981, #059669); color: white;"
            prefix = "✓"
        elif i == sum(steps_done_list):  # Current step
            style = "background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white;"
            prefix = "→"
        else:
            style = "background-color: #1e293b; color: #64748b;"
            prefix = "○"
        
        html_parts.append(f"""
        <span style="
            {style}
            padding: 8px 16px;
            border-radius: 20px;
            margin: 0 4px;
            display: inline-block;
            font-size: 12px;
            font-weight: 500;
        ">
            {prefix} {name}
        </span>
        """)
        
        if i < len(steps_done_list) - 1:
            html_parts.append("""
            <span style="color: #64748b; margin: 0 8px;">→</span>
            """)
    
    st.markdown(f"""
    <div style="
        text-align: center;
        margin: 20px 0;
        padding: 15px;
        background-color: #0f172a;
        border-radius: 10px;
        border: 1px solid #1e293b;
    ">
        {"".join(html_parts)}
    </div>
    """, unsafe_allow_html=True)