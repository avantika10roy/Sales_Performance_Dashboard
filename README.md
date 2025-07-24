# Sales Data Analysis Dashboard

## Overview

This project analyzes the sales performance of a specialty cookie company operating across multiple international markets. The company manufactures and sells six different cookie varieties in five key markets spanning Asia, Europe, and North America. The dataset covers a critical business period from September 2019 to December 2020, capturing 457 days of operational data during a time that includes both normal business operations and the COVID-19 pandemic impact.

## Dataset Structure

The application expects sales data with the following columns:

| Column | Description | Data Type |
|--------|-------------|-----------|
| **Date** | Transaction date (Excel serial date format) | Integer |
| **Country** | Country where the sale occurred | String |
| **Product** | Product type sold | String |
| **Units Sold** | Number of units sold | Integer |
| **Revenue** | Total revenue generated | Float |
| **Cost** | Total cost associated with the sale | Float |
| **Profit** | Profit earned (Revenue - Cost) | Float |

## Project Architecture

The application consists of eight main modules:

1. **Streamlit Web Application** (`server.py`) - Main web interface
2. **Data Loader** (`data_loader.py`) - Handles file loading
3. **Data Preprocessor** (`data_preprocessor.py`) - Data cleaning and validation
4. **Exploratory Data Analysis** (`data_eda.py`) - Analysis and visualizations
5. **COVID-19 Analysis** (`covid_analysis`) - Pre-COVID and COVID Time Period Analysis 
6. **Risk Analysis** (`risk_analysis`) - Understanding Concentration, Volatility and Profitability Risk
7. **Forecasting** (`forecasting`) - Forecasting future 90 days of Revenue Generation
8. **Model Fitting** (`model_fitting`) - Fitting various ensemble models, checking accuracy and plot actual vs predicted graph

## Features

### Data Processing
- **Multi-format Support**: CSV, XLS, and XLSX files
- **Automatic Data Cleaning**: Date formatting, duplicate removal, outlier handling
- **Data Quality Assessment**: Missing values, duplicates, and outlier detection
- **Secure File Handling**: Safe upload and processing

### Analytics & Visualizations
- **Key Performance Metrics**: Total revenue, profit margins, units sold
- **Performance Rankings**: Top countries and products by revenue
- **Comparative Analysis**: Revenue by country and product
- **Trend Analysis**: Time series sales patterns and seasonal decomposition
- **Advanced Charts**: Bubble plots, heatmaps, and distribution analysis

### COVID Analysis
- **Key Performance Metrics**: Average Revenue, Total Revenue and Profit Margin Pre-COVID and COVID
- **Comparative Analysis**: Average Sales Trend of Product/Country Pre-COVID and COVID

### Risk Analysis
- **Concentration Risk**: Pie Chart for Product and Country
- **Profitability Risk**: Histogram Plot on Profit Margin
- **Volatility Risk**: Line Chart on Revenue and Profit

### Forecasting
- **Forecasting 90 Days in Future**: Forecasting revenue generation possibility for next 90 days

### Model Fitting
- **Model Fit**: Four ensemble models fitted on the data
- **Accuracy Evaluation**: Evaluating accuracy of each model to find the best fit
- **Actual vs Predicted**: Line plot to understand the performance

### Dashboard Features
- **Interactive Interface**: Web-based dashboard with responsive design
- **Real-time Processing**: Instant analysis upon file upload
- **Professional Visualizations**: Plotly-powered interactive charts
- **Comprehensive Reports**: Complete data overview and statistical summaries

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step 1: Create Virtual Environment
```bash
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install streamlit pandas numpy plotly statsmodels werkzeug openpyxl xlrd
```

### Step 3: Create Directory Structure
```bash
mkdir -p data
mkdir -p src
```

### Step 4: File Organization
Ensure files are placed in correct locations:

project-root/
├── server.py
├── config.py
├── data/
├── requirements.txt
├── README.md
└── src/
    ├── data_loader.py
    ├── data_preprocessor.py
    ├── data_eda.py
    ├── covid_analysis.py
    ├── risk_analysis.py
    ├── forecasting.py
    └── model_fitting.py

### Step 5: Run Application
```bash
streamlit run server.py
```

Access the dashboard at `http://localhost:8501`

## Usage

1. **Launch Application**: Run the Streamlit application
2. **Upload Data**: Select your sales data file (CSV, XLS, or XLSX)
3. **View Dashboard**: Automatically generated analysis and visualizations
4. **Explore Insights**: Interactive charts and performance metrics
5. **Analyze COVID**: Performance metrics and trend analysis during and pre-COVID
6. **Risk Analysis**: Visualize possible risk through charts and graphs
7. **Forecasting**: Using forecasting models to predict future trend
8. **Model Fitting**: Checking the performance of models on the data

## System Requirements

### Supported File Formats
- CSV (Comma-separated values)
- XLS (Microsoft Excel 97-2003)
- XLSX (Microsoft Excel 2007+)

### Minimum Requirements
- **RAM**: 4GB (8GB recommended for large datasets)
- **Storage**: 100MB + space for uploaded files
- **Browser**: Chrome, Firefox, Edge, or Safari

## Dashboard Components

### Performance Metrics
- Total Revenue
- Average Revenue per Transaction
- Overall Profit Margin
- Total Units Sold
- Average Revenue Before and During COVID
- Average Units Sold Before and During COVID
- Profit Margin Before and During COVID

### Analysis Tables
- Top 3 Countries by Revenue
- Top 3 Products by Revenue
- Statistical Data Summary
- Data Quality Overview

### Visualizations
- Revenue by Country and Product (Bar Charts)
- Product Sales Distribution (Pie Chart)
- Sales Trends Over Time (Line Chart)
- Geographic Performance (Grouped Bar Chart)
- Revenue vs Profit Analysis (Bubble Plot)
- Profit Distribution (Heatmap)
- Seasonal Decomposition (Time Series)
- Average Sales Trend (Country/Product) for COVID and Pre-Covid Time (Line Chart)
- Concentration Risk (Pie Chart)
- Profitability Risk (Histogram Plot)
- Volatility Risk (Line Chart)
- Forecasting Trend (Line Chart)
- Accuracy Plot (Bar Chart)
- Actual vs Predicted (Line Chart)

## Data Processing Pipeline

The application automatically performs:

1. **Data Loading**: Secure file upload and format detection
2. **Quality Assessment**: Missing values, duplicates, and outlier detection
3. **Data Cleaning**: Date standardization, duplicate removal
4. **Outlier Handling**: IQR-based outlier treatment for key metrics
5. **Analysis Generation**: Comprehensive statistical and visual analysis
6. **Dashboard Rendering**: Interactive visualization display

## Troubleshooting

### Common Issues

**Module Import Errors**
- Verify virtual environment is activated
- Check file directory structure
- Ensure all dependencies are installed

**File Upload Problems**
- Check file format is supported (CSV, XLS, XLSX)
- Verify file permissions

**Memory Issues with Large Files**
- Increase system RAM allocation
- Process smaller data chunks
- Close unnecessary applications

**Browser Display Issues**
- Use supported browsers (Chrome, Firefox, Edge, Safari)
- Clear browser cache
- Check JavaScript is enabled

## Security Features

- Secure filename handling
- File type validation
- Upload directory isolation
- Input sanitization

## Technical Stack

- **Backend**: Streamlit (Python)
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly
- **Statistical Analysis**: Statsmodels
- **File Handling**: Werkzeug, OpenPyXL, XLRD
- **Model Performance**: Scikit-Learn

## License

This project is available under the MIT License.

## Support

For technical support or questions, create an issue in the project repository with detailed information about your setup and any error messages encountered.
