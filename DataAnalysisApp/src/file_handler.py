import pandas as pd
import numpy as np
import streamlit as st


def read_any_file(uploaded_file):
    """Read various file formats and return DataFrame, metadata, error code."""
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()

        if file_ext == 'csv':
            df = pd.read_csv(uploaded_file)
            return df, {}, None
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
            return df, {}, None
        elif file_ext == 'json':
            df = pd.read_json(uploaded_file)
            return df, {}, None
        elif file_ext == 'parquet':
            df = pd.read_parquet(uploaded_file)
            return df, {}, None
        else:
            return None, {'trace': f'File format not supported: {file_ext}'}, f'Unsupported format: {file_ext}'
    except Exception as e:
        return None, {'trace': str(e), 'notes': ['Error reading file']}, str(e)
