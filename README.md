# 📊 DataIQ — Professional Data Analytics App

A professional-grade data analytics web application built with Python and Streamlit.
Upload any CSV or Excel file and get instant data exploration, cleaning, and automated insights.

---

## 🚀 Features

- 📂 Upload CSV and Excel files
- 🔍 Deep data exploration with statistics and data types
- 🔢 Full value counts for any column
- 📦 GroupBy aggregation analysis
- 🧹 Inline data cleaning — fix missing values and duplicates column by column
- 📊 Automated data intelligence and insights
- 🔗 Correlation analysis between numeric columns
- 🚨 Outlier detection using IQR method
- 📋 Auto-generated executive summary report
- ⬇️ Download cleaned data as CSV or Excel

---

## 🛠️ Tech Stack

| Library      | Purpose                        |
|--------------|-------------------------------|
| Streamlit    | Web app framework              |
| Pandas       | Data manipulation              |
| NumPy        | Numerical operations           |
| Plotly       | Interactive charts             |
| Scikit-learn | Machine learning models        |
| OpenPyXL     | Excel file reading and writing |

---

## ⚙️ Setup Instructions

### 1. Clone or download this repository

```bash
git clone https://github.com/YOURUSERNAME/DataAnalysisApp.git
cd DataAnalysisApp
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate virtual environment

```bash
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the app

```bash
streamlit run app.py
```

### 6. Open in browser

The app will automatically open at:




## 📁 Supported File Types

| Format | Extension | What Happens |
|--------|-----------|-------------|
| CSV | `.csv` | Full data analysis |
| Excel | `.xlsx` `.xls` | Full data analysis, multi-sheet picker |
| JSON | `.json` | Parsed into table |
| Parquet | `.parquet` | Big data columnar format |
| TSV | `.tsv` `.txt` | Tab-separated data |
| SQLite | `.db` `.sqlite` | Table picker from database |
| SQL Script | `.sql` | Code viewer with syntax highlight |
| Python | `.py` | Code viewer with syntax highlight |
| PDF | `.pdf` | Text extractor |
| Word | `.docx` | Paragraph + table extractor |
| Image | `.png` `.jpg` | Image viewer with metadata |
| Power BI | `.pbix` | Archive inspector + page names |
| XML | `.xml` | Parsed into table |
| ODS | `.ods` | LibreOffice spreadsheet |
| Feather | `.feather` | Fast columnar format |