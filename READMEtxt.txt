# 🚀 Galactic Data Forge – Senior Data Scientist Workbench

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-deployed-url.com)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**A professional, end‑to‑end data science platform**  
Upload, explore, clean, preprocess, test hypotheses, select features, train classical ML models, build deep neural networks, and export results – all in one beautifully dark, interactive web app.

---

## 📖 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture & Workflow](#️-architecture--workflow)
- [⚙️ Installation](#️-installation)
- [🚀 Running the App](#-running-the-app)
- [📊 Detailed Step‑by‑Step Guide](#-detailed-step-by-step-guide)
  - [1. Upload Data](#1-upload-data)
  - [2. EDA & Insights](#2-eda--insights)
  - [3. Data Cleaning](#3-data-cleaning)
  - [4. Statistical Summary](#4-statistical-summary)
  - [5. Hypothesis Testing](#5-hypothesis-testing)
  - [6. Preprocessing (Scaling & Encoding)](#6-preprocessing-scaling--encoding)
  - [7. Intelligent Feature Selection](#7-intelligent-feature-selection)
  - [8. Classical Machine Learning](#8-classical-machine-learning)
  - [9. Deep Learning](#9-deep-learning)
  - [10. Batch Prediction](#10-batch-prediction)
  - [11. Export Processed Data](#11-export-processed-data)
- [📈 Health Metrics & Model Reports](#-health-metrics--model-reports)
- [🛠️ Configuration & Customisation](#️-configuration--customisation)
- [🐛 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

| Category | Capabilities |
|----------|--------------|
| **Data Loading** | CSV, Excel, JSON, Parquet (auto‑detect) |
| **EDA** | Interactive histograms, boxplots, correlation heatmap, missing values chart, column info, skewness & kurtosis |
| **Cleaning** | Remove duplicates, drop columns with >50% missing, impute numeric (median/mean/KNN), fill categorical with mode |
| **Statistics** | Descriptive stats, skewness, kurtosis |
| **Hypothesis Testing** | 20+ tests: normality (Shapiro, D’Agostino, Anderson‑Darling), t‑tests (one‑sample, independent, paired), ANOVA, Chi‑square, Fisher’s exact, correlation (Pearson/Spearman/Kendall), variance equality (Levene/Bartlett) – **user‑selectable significance level α** |
| **Preprocessing** | Scaling (Standard, MinMax, Robust), encoding (Label, One‑Hot, Target) |
| **Feature Selection** | Automatic ranking using **Mutual Information + Correlation** – recommends top 10 features |
| **Classical ML** | Classification, Regression, Clustering – **all major algorithms** (Logistic, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVM, SVR, Linear/Ridge/Lasso/ElasticNet, KMeans, DBSCAN, Agglomerative, GMM). **Hyperparameter tuning**, **cross‑validation**, **learning curves**, **confusion matrix**, **residual plots**, **feature importance**, **model health report** (overfitting/underfitting diagnosis). |
| **Deep Learning** | ANN, CNN 1D, RNN (LSTM), Transformer – **early stopping**, **training curves**, **classification report**, **confusion matrix**, **residual plots**. |
| **Health Metrics** | After every step: rows, columns, missing %, duplicates %, completeness score, grade |
| **Export** | Download processed dataset as CSV |
| **UI** | Professional dark theme (GitHub Dark), responsive, interactive Plotly charts |

---

## 🏗️ Architecture & Workflow

The application follows a **linear 11‑step pipeline**, preserving data in `st.session_state` so you can go back and forth. Each step shows **health metrics** and **explanations** of what is happening.
Upload → EDA → Clean → Stats → Hypothesis → Preprocess → Feature Select → ML → DL → Predict → Export

text

- **Session persistence** – your data stays until you close the browser.
- **Safe copies** – original raw data is never modified.
- **Modular functions** – each step is a separate, testable function.

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) virtual environment

### 1. Clone the repository
```bash
git clone https://github.com/your-username/DataAnalysisApp.git
cd DataAnalysisApp
2. Create and activate a virtual environment (recommended)
bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
3. Install dependencies
bash
pip install -r requirements.txt
If you don’t have requirements.txt, install core packages:

bash
pip install streamlit pandas numpy plotly scikit-learn scipy statsmodels
# Optional (for advanced models):
pip install xgboost lightgbm catboost tensorflow
🚀 Running the App
bash
streamlit run app.py
Your browser will open at http://localhost:8501.
Use the sidebar to navigate through the 11 steps.

📊 Detailed Step‑by‑Step Guide
1. Upload Data
What it does: Loads a CSV, Excel, JSON, or Parquet file.

What you see: First 20 rows, automatic data insights (numeric/categorical columns, missing cells, skewness, cardinality, suggestions).

Health metrics: rows, columns, missing %, duplicates %, completeness, grade.

2. EDA & Insights
Interactive preview: filter columns to display.

Column info table: type, non‑null count, null %, unique, skewness.

Distribution plot: for any numeric column – histogram + boxplot.

Correlation heatmap: if ≥2 numeric columns.

Missing values bar chart.

Explanations for every chart.

3. Data Cleaning
Remove duplicate rows.

Drop columns with >50% missing values.

Numeric imputation: Median, Mean, KNN (k=5).

Categorical imputation: fill missing with mode.

After cleaning, health metrics update automatically.

4. Statistical Summary
Descriptive statistics (count, mean, std, min, quartiles, max).

Skewness & Kurtosis table.

5. Hypothesis Testing
Choose α (significance level): 0.01, 0.05, or 0.10.

Test categories:

Normality: Shapiro‑Wilk, D’Agostino‑Pearson, Anderson‑Darling.

One‑sample: t‑test, Wilcoxon signed‑rank.

Two‑sample independent: t‑test (equal var), Welch t‑test, Mann‑Whitney U.

Paired: paired t‑test, Wilcoxon signed‑rank.

ANOVA (with Tukey HSD post‑hoc for ≤5 groups).

Categorical association: Chi‑square, Fisher’s exact (2x2).

Correlation: Pearson, Spearman, Kendall.

Variance equality: Levene, Bartlett.

Results show test statistic, p‑value, and interpretation (reject/fail to reject H₀).

6. Preprocessing (Scaling & Encoding)
Scaling: StandardScaler, MinMaxScaler, RobustScaler.

Encoding: Label, One‑Hot (drop first), Target (mean encoding).

After applying, health metrics update.

7. Intelligent Feature Selection
Select target column and problem type (classification/regression).

Auto‑recommend – combines Mutual Information (non‑linear) and Correlation (linear) to rank features. Displays a bar chart.

Adjust feature set manually with multiselect.

Proceed to modeling when ready.

8. Classical Machine Learning
Choose task: Classification, Regression, or Clustering.

Choose algorithm (list depends on task; shows only installed libraries).

Hyperparameter tuning (optional expander):

Number of estimators, max depth, learning rate, L1/L2 regularization.

Number of clusters (for clustering).

Cross‑validation folds (2–10).

After training:

Test score (accuracy / R²) and cross‑validation mean score ± std.

Learning curve.

Confusion matrix (classification) / residual plot (regression).

Feature importance (for tree‑based models).

Model health report – diagnoses overfitting/underfitting by comparing train vs test scores and CV variability.

9. Deep Learning
Automatic task detection – if target has >20 unique numeric values, defaults to Regression.

Architecture: ANN, CNN 1D, RNN (LSTM), Transformer.

Epochs, batch size, validation split, early stopping patience.

Training curves (loss and metric) in real time.

Evaluation:

Classification: test accuracy, classification report, confusion matrix.

Regression: test MAE, residual plot, R² score.

(Requires TensorFlow; if not installed, shows an informative error.)

10. Batch Prediction
Placeholder for future feature – upload new data and apply a saved model.

11. Export Processed Data
Download the current processed DataFrame as CSV.

📈 Health Metrics & Model Reports
Health Metrics (after each step)
Metric	Explanation
Rows	Number of observations
Columns	Number of features
Missing %	Percentage of all cells that are missing
Duplicates %	Percentage of duplicate rows
Completeness	100% − Missing %
Grade	Excellent (≥95%), Good (≥80%), Poor (<80%)
Model Health Report (ML step)
CV Mean Score – average performance across k‑fold cross‑validation.

CV Std Dev – variability; high std suggests instability.

Train Score – model performance on training data.

Test Score – performance on unseen test data.

Overfitting diagnosis: large gap between train and test → overfitting; both low → underfitting; close values → good balance.

