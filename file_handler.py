import pandas as pd
import numpy as np
import streamlit as st
import io
import os
import zipfile
import sqlite3
import traceback

# ── READ ANY FILE AND RETURN A DATAFRAME + METADATA ──────────
# Returns: (dataframe_or_None, metadata_dict, error_string_or_None)

def read_any_file(file):

    name    = file.name.lower()
    meta    = {'filename': file.name, 'filetype': None, 'notes': []}
    raw     = file.read()
    file.seek(0)

    try:

        # ── CSV ───────────────────────────────────────────────
        if name.endswith('.csv'):
            meta['filetype'] = 'CSV'
            try:
                df = pd.read_csv(io.BytesIO(raw))
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(raw), encoding='latin-1')
                meta['notes'].append('Used latin-1 encoding (file had special characters)')
            return df, meta, None

        # ── EXCEL XLSX ────────────────────────────────────────
        elif name.endswith('.xlsx'):
            meta['filetype'] = 'Excel (.xlsx)'
            xl   = pd.ExcelFile(io.BytesIO(raw), engine='openpyxl')
            meta['sheets'] = xl.sheet_names
            if len(xl.sheet_names) == 1:
                df = xl.parse(xl.sheet_names[0])
                meta['notes'].append(f"Single sheet: '{xl.sheet_names[0]}'")
                return df, meta, None
            else:
                meta['notes'].append(f"Multiple sheets found: {xl.sheet_names}")
                meta['multi_sheet'] = True
                meta['excel_file']  = xl
                return None, meta, 'MULTI_SHEET'

        # ── OLD EXCEL XLS ─────────────────────────────────────
        elif name.endswith('.xls'):
            meta['filetype'] = 'Excel (.xls)'
            xl   = pd.ExcelFile(io.BytesIO(raw), engine='xlrd')
            meta['sheets'] = xl.sheet_names
            if len(xl.sheet_names) == 1:
                df = xl.parse(xl.sheet_names[0])
                return df, meta, None
            else:
                meta['multi_sheet'] = True
                meta['excel_file']  = xl
                return None, meta, 'MULTI_SHEET'

        # ── JSON ──────────────────────────────────────────────
        elif name.endswith('.json'):
            meta['filetype'] = 'JSON'
            try:
                df = pd.read_json(io.BytesIO(raw))
            except ValueError:
                # Try reading as JSON lines format
                df = pd.read_json(io.BytesIO(raw), lines=True)
                meta['notes'].append('Parsed as JSON Lines format')
            return df, meta, None

        # ── PARQUET ───────────────────────────────────────────
        elif name.endswith('.parquet'):
            meta['filetype'] = 'Parquet'
            df = pd.read_parquet(io.BytesIO(raw))
            meta['notes'].append('Big data columnar format — all columns loaded')
            return df, meta, None

        # ── TSV (Tab Separated) ───────────────────────────────
        elif name.endswith('.tsv') or name.endswith('.txt'):
            meta['filetype'] = 'TSV / TXT'
            try:
                df = pd.read_csv(io.BytesIO(raw), sep='\t')
                meta['notes'].append('Parsed as tab-separated values')
            except Exception:
                df = pd.read_csv(io.BytesIO(raw), sep=None, engine='python')
                meta['notes'].append('Auto-detected separator')
            return df, meta, None

        # ── SQLITE DATABASE ───────────────────────────────────
        elif name.endswith('.db') or name.endswith('.sqlite') or name.endswith('.sqlite3'):
            meta['filetype'] = 'SQLite Database'
            tmp_path = f"_temp_{file.name}"
            with open(tmp_path, 'wb') as f:
                f.write(raw)
            try:
                conn   = sqlite3.connect(tmp_path)
                tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
                meta['tables'] = tables['name'].tolist()
                meta['notes'].append(f"Tables found: {meta['tables']}")
                if len(meta['tables']) == 0:
                    return None, meta, 'NO_TABLES'
                meta['multi_table'] = True
                meta['db_path']     = tmp_path
                meta['db_conn']     = conn
                return None, meta, 'MULTI_TABLE'
            except Exception as e:
                os.remove(tmp_path)
                return None, meta, str(e)

        # ── SQL TEXT FILE ─────────────────────────────────────
        elif name.endswith('.sql'):
            meta['filetype'] = 'SQL Script'
            try:
                sql_text = raw.decode('utf-8')
            except UnicodeDecodeError:
                sql_text = raw.decode('latin-1')
            meta['sql_text'] = sql_text
            meta['notes'].append('SQL script displayed — not executable data')
            return None, meta, 'SQL_SCRIPT'

        # ── PYTHON FILE ───────────────────────────────────────
        elif name.endswith('.py'):
            meta['filetype'] = 'Python Script'
            try:
                py_text = raw.decode('utf-8')
            except UnicodeDecodeError:
                py_text = raw.decode('latin-1')
            meta['py_text'] = py_text
            meta['notes'].append('Python script displayed — not data')
            return None, meta, 'PYTHON_SCRIPT'

        # ── PDF ───────────────────────────────────────────────
        elif name.endswith('.pdf'):
            meta['filetype'] = 'PDF Document'
            try:
                import PyPDF2
                reader   = PyPDF2.PdfReader(io.BytesIO(raw))
                n_pages  = len(reader.pages)
                text     = ''
                for i, page in enumerate(reader.pages):
                    try:
                        text += f"\n--- Page {i+1} ---\n"
                        text += page.extract_text() or ''
                    except Exception:
                        text += f"\n[Could not extract text from page {i+1}]\n"
                meta['pdf_text'] = text
                meta['n_pages']  = n_pages
                meta['notes'].append(f'{n_pages} pages extracted')

                # Try to find tables in the text
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                meta['pdf_lines'] = lines
                return None, meta, 'PDF_FILE'
            except ImportError:
                return None, meta, 'PDF_NO_LIB'

        # ── WORD DOCX ─────────────────────────────────────────
        elif name.endswith('.docx'):
            meta['filetype'] = 'Word Document (.docx)'
            try:
                from docx import Document
                doc      = Document(io.BytesIO(raw))
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                tables_data = []
                for table in doc.tables:
                    rows = []
                    for row in table.rows:
                        rows.append([cell.text.strip() for cell in row.cells])
                    if rows:
                        try:
                            tdf = pd.DataFrame(rows[1:], columns=rows[0])
                            tables_data.append(tdf)
                        except Exception:
                            pass
                meta['doc_paragraphs'] = paragraphs
                meta['doc_tables']     = tables_data
                meta['notes'].append(f'{len(paragraphs)} paragraphs, {len(tables_data)} tables found')
                if tables_data:
                    meta['notes'].append('Tables extracted and available as DataFrames')
                    return tables_data[0], meta, None
                return None, meta, 'DOCX_NO_TABLE'
            except ImportError:
                return None, meta, 'DOCX_NO_LIB'

        # ── IMAGE ─────────────────────────────────────────────
        elif name.endswith(('.png','.jpg','.jpeg','.bmp','.gif','.webp')):
            meta['filetype'] = 'Image'
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(raw))
                meta['img_size']   = img.size
                meta['img_mode']   = img.mode
                meta['img_object'] = img
                meta['notes'].append(f'Image: {img.size[0]}×{img.size[1]} px, mode {img.mode}')
                return None, meta, 'IMAGE_FILE'
            except ImportError:
                return None, meta, 'IMAGE_NO_LIB'

        # ── POWER BI PBIX ─────────────────────────────────────
        elif name.endswith('.pbix'):
            meta['filetype'] = 'Power BI (.pbix)'
            try:
                # pbix is a ZIP archive — we open it and list contents
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    file_list = z.namelist()
                    meta['pbix_contents'] = file_list
                    meta['notes'].append(f'{len(file_list)} internal files found in .pbix archive')

                    # Try to extract the DataModel metadata
                    data_files = [f for f in file_list if 'DataModel' in f or 'data' in f.lower()]
                    meta['pbix_data_files'] = data_files

                    # Try reading Connections file for source info
                    if 'Connections' in file_list:
                        try:
                            conn_bytes = z.read('Connections')
                            meta['pbix_connections'] = conn_bytes.decode('utf-8', errors='replace')
                        except Exception:
                            pass

                    # Try reading Report/Layout for page names
                    if 'Report/Layout' in file_list:
                        try:
                            import json
                            layout_bytes = z.read('Report/Layout')
                            layout       = json.loads(layout_bytes.decode('utf-8', errors='replace'))
                            sections     = layout.get('sections', [])
                            page_names   = [s.get('displayName','') for s in sections]
                            meta['pbix_pages'] = page_names
                            meta['notes'].append(f"Report pages: {page_names}")
                        except Exception:
                            pass

                return None, meta, 'PBIX_FILE'
            except zipfile.BadZipFile:
                return None, meta, 'PBIX_CORRUPT'

        # ── XML ───────────────────────────────────────────────
        elif name.endswith('.xml'):
            meta['filetype'] = 'XML'
            try:
                df = pd.read_xml(io.BytesIO(raw))
                meta['notes'].append('Parsed XML into tabular format')
                return df, meta, None
            except Exception as e:
                return None, meta, f'XML parse error: {e}'

        # ── ODS (LibreOffice Spreadsheet) ─────────────────────
        elif name.endswith('.ods'):
            meta['filetype'] = 'ODS Spreadsheet'
            df = pd.read_excel(io.BytesIO(raw), engine='odf')
            return df, meta, None

        # ── FEATHER ───────────────────────────────────────────
        elif name.endswith('.feather'):
            meta['filetype'] = 'Feather'
            df = pd.read_feather(io.BytesIO(raw))
            return df, meta, None

        # ── UNKNOWN FILE TYPE ─────────────────────────────────
        else:
            meta['filetype'] = 'Unknown'
            return None, meta, 'UNKNOWN_TYPE'

    except Exception as e:
        meta['error_trace'] = traceback.format_exc()
        return None, meta, str(e)
    