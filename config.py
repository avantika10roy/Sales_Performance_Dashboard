import plotly.express as px

class Config:

    HEIGHT           = 500
    WEIGHT           = 800
    WINDOW           = 10
    DATA_PATH        = 'data/sales_data.csv'
    NUM_COLUMN       = ['Units Sold', 'Revenue', 'Cost', 'Profit']
    Z_THRESHOLD      = 3
    FUTURE_STEPS     = 10
    COLOR_PALETTE    = px.colors.qualitative.Pastel
    EXCEL_START_DATE = '1900-01-01'

