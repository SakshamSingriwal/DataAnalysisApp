# 📊 DataAnalysisApp — Professional Data Analytics & ML Platform

A comprehensive, production-ready data analytics and machine learning platform built with **Python** and **Streamlit**. Upload any data file and perform exploratory data analysis, data preprocessing, statistical testing, machine learning modeling, and deep learning—all in an intuitive web interface.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Features

### 📂 **File Handling**
- Support for 20+ file formats: CSV, Excel, JSON, Parquet, SQL databases, PDFs, images, and more
- Automatic file type detection and parsing
- Multi-sheet Excel support
- Database table picker for SQLite

### 🔍 **Exploratory Data Analysis (EDA)**
- Comprehensive data profiling and statistics
- Data type detection and validation
- Missing values analysis and visualization
- Duplicate detection and removal
- Outlier detection using IQR method
- Correlation and covariance analysis
- Automated data quality scoring

### 🧹 **Data Preprocessing**
- **Encoding**: One-Hot, Ordinal, Label, Binary encoding
- **Scaling**: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
- **Imputation**: Simple imputation, KNN imputation
- **Feature Engineering**: Feature extraction and creation
- **Batch operations**: Process multiple columns simultaneously

### 📊 **Statistical Analysis**
- Hypothesis testing suite:
  - Normality tests (Shapiro-Wilk, D'Agostino-Pearson)
  - t-tests (one-sample, two-sample, paired, Welch)
  - ANOVA (one-way, repeated measures)
  - Chi-square test of independence
  - Non-parametric tests (Mann-Whitney U, Kruskal-Wallis, Wilcoxon)
  - Correlation tests (Pearson, Spearman, Kendall)
  - Variance tests (Levene, Bartlett)
- Customizable significance levels
- P-value interpretation and recommendations

### 🤖 **Machine Learning Models**

#### Classification
- Logistic Regression
- Decision Trees
- Random Forests
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- AdaBoost, Extra Trees

#### Regression
- Linear & Regularized Regression (Ridge, Lasso, ElasticNet)
- Decision Trees & Random Forests
- Gradient Boosting
- Support Vector Regression
- K-Nearest Neighbors
- Huber Regressor

#### Clustering
- K-Means
- DBSCAN
- Agglomerative Clustering
- Spectral Clustering
- Gaussian Mixture Models

#### Feature Importance & Interpretation
- SHAP values for model explainability
- Feature importance rankings
- Permutation importance

### 🧠 **Deep Learning**
- Neural network architecture builder
- Dropout and batch normalization support
- TensorFlow/Keras integration
- Custom layer configuration
- Training visualization and performance metrics

### 📈 **Visualization & Reporting**
- Interactive plots with Plotly
- Real-time data visualization
- Model performance dashboards
- Confusion matrices and classification reports
- ROC curves and precision-recall curves
- Automated executive summary reports

### 💾 **Data Export**
- Download processed data as CSV or Excel
- Export model predictions
- Save trained models
- Generate analysis reports

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM, CatBoost |
| **Deep Learning** | TensorFlow, Keras |
| **Statistical Analysis** | SciPy, Statsmodels |
| **Model Explainability** | SHAP |
| **File I/O** | OpenPyXL, PyArrow, python-docx, PyPDF2 |

---

## 📋 Project Structure

```
DataAnalysisApp/
├── app.py                 # Main Streamlit application
├── file_handler.py        # Universal file reader (20+ formats)
├── utils.py               # Utility functions and UI components
├── preprocessing.py       # Data preprocessing pipeline
├── ml_models.py           # Machine learning models implementation
├── deep_learning.py       # Deep learning models and neural networks
├── stats_tests.py         # Statistical hypothesis testing suite
├── merge_helpers.py       # Data merging and joining utilities
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
└── __pycache__/           # Python cache (auto-generated)
```

### Module Descriptions

**app.py**
- Main entry point for the Streamlit application
- Orchestrates UI layout and navigation
- Manages session state and data flow
- Integrates all submodules

**file_handler.py**
- Supports 20+ file formats automatically
- Handles encoding detection and error recovery
- Returns DataFrames with metadata

**utils.py**
- Reusable utility functions
- Custom UI components (info/success/warning boxes)
- Data quality scoring
- Memory size formatting

**preprocessing.py**
- Feature scaling and normalization
- Categorical encoding (One-Hot, Ordinal, Label)
- Missing value imputation (Simple, KNN)
- Feature engineering tools

**ml_models.py**
- Classification models (12+ algorithms)
- Regression models (10+ algorithms)
- Clustering algorithms (5+ variants)
- Cross-validation and hyperparameter tuning
- Model evaluation metrics

**deep_learning.py**
- Neural network architecture builder
- Dense layers with various activations
- Dropout and batch normalization
- Training with early stopping
- Performance visualization

**stats_tests.py**
- Comprehensive hypothesis testing suite
- Parametric and non-parametric tests
- P-value interpretation
- Statistical recommendations

**merge_helpers.py**
- Multi-file merging utilities
- Join operations (inner, outer, left, right)
- Data validation and conflict resolution

---

## ⚙️ Installation & Setup

### Prerequisites
- **Python 3.8 or higher**
- **pip** (Python package manager)
- **Virtual Environment** (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/SakshamSingriwal/DataAnalysisApp.git
cd DataAnalysisApp
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

### Step 5: Access the App

The application will automatically open in your browser at:
```
http://localhost:8501
```

---

## 🚀 Usage Guide

### Basic Workflow

1. **Upload Data**: Click "Upload File" and select your data file (CSV, Excel, JSON, etc.)
2. **Explore**: View data summary, statistics, distributions, and correlations
3. **Clean**: Handle missing values, remove duplicates, detect outliers
4. **Preprocess**: Encode categorical features, scale numerical features
5. **Analyze**: Run statistical tests and hypothesis testing
6. **Model**: Train and compare machine learning models
7. **Predict**: Generate predictions on new data
8. **Export**: Download processed data and results

### Advanced Features

#### Custom Machine Learning Pipeline
1. Navigate to "Machine Learning" tab
2. Select problem type (Classification/Regression/Clustering)
3. Choose target variable
4. Select models to train
5. Configure hyperparameters (optional)
6. View performance metrics and comparisons
7. Export best model predictions

#### Deep Learning Models
1. Go to "Deep Learning" section
2. Define neural network architecture
3. Configure training parameters
4. Monitor training progress with real-time charts
5. Evaluate on test set
6. Make predictions

#### Statistical Testing
1. Select "Statistical Tests" tab
2. Choose test category (normality, t-tests, ANOVA, etc.)
3. Select columns and parameters
4. View results with interpretation
5. Export test report

---

## 📁 Supported File Formats

| Format | Extension | Status |
|--------|-----------|--------|
| CSV | `.csv` | ✅ Full Support |
| Excel | `.xlsx`, `.xls` | ✅ Full Support |
| JSON | `.json` | ✅ Full Support |
| Parquet | `.parquet` | ✅ Full Support |
| TSV/TXT | `.tsv`, `.txt` | ✅ Full Support |
| SQLite | `.db`, `.sqlite` | ✅ Full Support |
| Feather | `.feather` | ✅ Full Support |
| PDF | `.pdf` | ✅ Text Extraction |
| Word | `.docx` | ✅ Table Extraction |
| XML | `.xml` | ✅ Parsing |
| ODS | `.ods` | ✅ LibreOffice Support |
| Images | `.png`, `.jpg` | ✅ Metadata |
| SQL | `.sql` | ✅ Code Viewer |
| Python | `.py` | ✅ Code Viewer |

---

## 🔧 Configuration

### Customizing the Application

Edit `app.py` to customize:
- Page layout and theme
- Default settings
- UI colors and styling
- Sidebar configuration

### Environment Variables

Create a `.env` file (optional):
```bash
STREAMLIT_LOGGER_LEVEL=error
TF_CPP_MIN_LOG_LEVEL=2  # Suppress TensorFlow warnings
```

---

## 📊 Performance Optimization

- **Data Caching**: Streamlit caches file uploads and computations
- **Lazy Loading**: Models load only when selected
- **Efficient Algorithms**: Uses optimized scikit-learn implementations
- **GPU Support**: TensorFlow can utilize GPU if available

---

## 🐛 Troubleshooting

### Issue: TensorFlow/Keras not found
**Solution**: Install explicitly
```bash
pip install tensorflow keras
```

### Issue: Large file upload fails
**Solution**: Increase Streamlit server limits in `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200
```

### Issue: Port 8501 already in use
**Solution**: Specify different port
```bash
streamlit run app.py --server.port 8502
```

---

## 📝 Module Integration Guide

All modules are tightly integrated through the main `app.py`:

1. **File Handler** → Reads uploaded files
2. **Utils** → Scores data quality
3. **Preprocessing** → Cleans and transforms data
4. **ML Models** → Trains and evaluates models
5. **Deep Learning** → Builds neural networks
6. **Stats Tests** → Performs statistical analysis
7. **Merge Helpers** → Combines multiple datasets

Each module maintains consistent:
- Error handling and logging
- UI styling and components
- Data format and validation
- Performance optimization

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Saksham Singriwal**
- GitHub: [@SakshamSingriwal](https://github.com/SakshamSingriwal)
- Email: contact@example.com

---

## 📞 Support & Feedback

- 📧 Email: support@example.com
- 🐙 GitHub Issues: [Report a bug](https://github.com/SakshamSingriwal/DataAnalysisApp/issues)
- 💬 Discussions: [Ask a question](https://github.com/SakshamSingriwal/DataAnalysisApp/discussions)

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Machine learning with [Scikit-learn](https://scikit-learn.org/)
- Deep learning with [TensorFlow/Keras](https://www.tensorflow.org/)
- Statistical analysis with [SciPy](https://scipy.org/)
- Model interpretation with [SHAP](https://shap.readthedocs.io/)

---

## 📊 Screenshots & Demo

*Add screenshots of the application interface here*

---

**Happy Analyzing! 🚀**