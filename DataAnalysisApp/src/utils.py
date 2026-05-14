import pandas as pd
import numpy as np


def compute_quality(df):
    """Compute data quality score for a DataFrame."""
    if df.empty:
        return 0.0, 'F', 100.0, 0.0

    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0

    duplicate_rows = df.duplicated().sum()
    dup_pct = (duplicate_rows / len(df) * 100) if len(df) > 0 else 0

    quality_score = max(0, 100 - missing_pct - (dup_pct * 0.5))

    if quality_score >= 90:
        grade = 'A'
    elif quality_score >= 80:
        grade = 'B'
    elif quality_score >= 70:
        grade = 'C'
    elif quality_score >= 60:
        grade = 'D'
    else:
        grade = 'F'

    return quality_score, grade, missing_pct, dup_pct


def fmt_bytes(num_bytes):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"
