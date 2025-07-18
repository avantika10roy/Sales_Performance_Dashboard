# Sales Performance Analysis & BI Dashboard

## 🚀 Overview

This interactive **Sales Performance Dashboard** provides end-to-end analytics for a specialty product company operating across multi-country markets. Built with **Python** and **Flask**, it delivers:

- Data Overview
- Statistical Summary
- Data Quality Metrics
- KPI Metrics
- Top 3 Countries w.r.t Sales
- Top 3 Performing Products
- Different Kind of Analytical Charts and Plots
---

## 📝 Business Objectives

✅ Identify top-performing **markets** and **products**
✅ Deliver **actionable insights** through a fully automated **executive summary**

---
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

The application consists of four main modules:

1. **Flask Web Application** (`app.py`) - Main web interface
2. **Data Loader** (`data_loader.py`) - Handles file loading
3. **Data Preprocessor** (`data_preprocessor.py`) - Data cleaning and validation
4. **Exploratory Data Analysis** (`data_eda.py`) - Analysis and visualizations

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
pip install flask pandas numpy plotly statsmodels werkzeug openpyxl xlrd
```

### Step 3: Create Directory Structure
```bash
mkdir -p static/uploads
mkdir -p templates
mkdir -p src
```

### Step 4: File Organization
Ensure files are placed in correct locations:
- `app.py` - Root directory
- `src/data_loader.py` - Data loading module
- `src/data_preprocessor.py` - Data preprocessing
- `src/data_eda.py` - Analysis module
- `templates/` - HTML templates
- `static/uploads/` - File upload directory

### Step 5: Run Application
```bash
python app.py
```

Access the dashboard at `http://localhost:5000`

## Usage

1. **Launch Application**: Run the Flask application
2. **Upload Data**: Select your sales data file (CSV, XLS, or XLSX)
3. **View Dashboard**: Automatically generated analysis and visualizations
4. **Explore Insights**: Interactive charts and performance metrics

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

## Data Processing Pipeline

The application automatically performs:

1. **Data Loading**: Secure file upload and format detection
2. **Quality Assessment**: Missing values, duplicates, and outlier detection
3. **Data Cleaning**: Date standardization, duplicate removal
4. **Outlier Handling**: Z-Score-based outlier treatment for key metrics
5. **Analysis Generation**: Comprehensive statistical and visual analysis
6. **Dashboard Rendering**: Interactive visualization display

## Troubleshooting

### Common Issues

**Module Import Errors**
- Verify virtual environment is activated
- Check file directory structure
- Ensure all dependencies are installed

**File Upload Problems**
- Confirm `static/uploads/` directory exists
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

- **Backend**: Flask (Python)
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly
- **Statistical Analysis**: Statsmodels
- **File Handling**: Werkzeug

## License

This project is available under the MIT License.
