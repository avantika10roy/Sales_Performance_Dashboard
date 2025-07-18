import os
from src.data_loader import DataLoader
from werkzeug.utils import secure_filename
from src.data_preprocessor import DataInspection
from src.data_eda import ExploratoryDataAnalysis
from flask import Flask, render_template, request

########################## APP INITIALIZATION ###############################

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Allowed File Extensions
ALLOWED_EXTENSIONS = ['csv', 'xls', 'xlsx']

# Ensure Upload Directory Exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok = True)

# Function to allows specific files to be uploaded
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# handling multiple HTTP methods
@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Secure the filepath
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the file
            file.save(filepath)

            # Load Data
            loader = DataLoader()
            data   = loader.load_data(input_data = filepath)

            # Summary Statistics (KPI)
            preprocessor = DataInspection(input_data = data)
            data_columns = preprocessor.col_names()
            rows_columns = preprocessor.data_shape()
            missing_val  = preprocessor.find_null()
            duplicates   = preprocessor.find_duplicate()
            outliers     = preprocessor.check_outliers()

            # Preprocess Data
            data  = preprocessor.change_date_format()
            data  = preprocessor.sort_data()
            data   = preprocessor.remove_duplicates()
            data = preprocessor.handle_outliers(window      = 10,
                                                column_name = 'Revenue')
            data = preprocessor.handle_outliers(window      = 10,
                                                column_name = 'Units Sold')
            data = preprocessor.handle_outliers(window      = 10,
                                                column_name = 'Profit')
            data = preprocessor.handle_outliers(window      = 10,
                                                column_name = 'Cost')
                
            processed_data = data

            # Create Plots for Visualization
            eda = ExploratoryDataAnalysis(input_data = processed_data)
            description              = preprocessor.data_description()                              # Table
            revenue_sum              = eda.total_revenue()                                          # Row 1a
            revenue_avg              = eda.average_revenue()                                        # Row 1b
            profit_margin            = eda.profit_margin()                                          # Row 1c
            total_units              = eda.total_units()                                            # Row 1d
            top_countries            = eda.top_3_countries()                                        # Tab Left
            top_products             = eda.top_3_products()                                         # Tab Right
            sale_by_country          = eda.barplot_total_sale_by_column(column_name = 'Country')    # Left
            sale_by_product          = eda.barplot_total_sale_by_column(column_name = 'Product')    # Right
            product_sale_percentage  = eda.pieplot_for_product_sale()                               # Left
            product_per_country_sale = eda.groupedbar_product_over_country()                        # Right
            sales_trend              = eda.sales_trend_per_product()                                # Center
            bubble_plot              = eda.simple_bubble_plot()                                     # Left                              # Right
            heatmap_plot             = eda.heatmap_for_product_profit()                             # center
            seasonal_decomposition   = eda.decompose_revenue()                                      # Center

            return render_template('dashboard.html',
                                   data_overview_tables     = processed_data.head(10).to_html(classes = 'table styled-table', index = False),
                                   data_columns             = list(data_columns),
                                   rows_columns             = rows_columns,
                                   missing_val              = dict(missing_val),
                                   duplicates               = duplicates,
                                   outliers                 = outliers,
                                   description              = description.to_html(classes = 'table statistical-summary', index = False),                              # Table
                                   revenue_sum              = revenue_sum,                                          # Row 1a
                                   revenue_avg              = revenue_avg,                                        # Row 1b
                                   profit_margin            = profit_margin,                                          # Row 1c
                                   total_units              = total_units,                                            # Row 1d
                                   top_countries            = top_countries.to_html(classes = 'table statistical-summary', index = False),                                        # Tab Left
                                   top_products             = top_products.to_html(classes = 'table statistical-summary', index = False),                                         # Tab Right
                                   sale_by_country          = sale_by_country,    # Left
                                   sale_by_product          = sale_by_product,    # Right
                                   product_sale_percentage  = product_sale_percentage,                               # Left
                                   product_per_country_sale = product_per_country_sale,                        # Right
                                   sales_trend              = sales_trend,                                # Center
                                   bubble_plot              = bubble_plot,                                     # Left
                                   heatmap_plot             = heatmap_plot,                             # center
                                   seasonal_decomposition   = seasonal_decomposition)
        
    return render_template('base.html')

if __name__ == '__main__':
    app.run(debug = True)