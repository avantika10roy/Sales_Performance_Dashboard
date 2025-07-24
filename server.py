import pandas as pd
import streamlit as st
from config import Config
from src.data_loader import DataLoader
from src.forecasting import Forecasting
from src.model_fitting import ModelFitter
from src.risk_analysis import RiskAnalysis
from werkzeug.utils import secure_filename
from streamlit import session_state as state
from src.covid_analysis import CovidAnalysis
from src.data_preprocessor import DataInspection
from src.data_eda import ExploratoryDataAnalysis


st.set_page_config(page_title="Sales Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")
################################################################## SIDE BAR CONFIGURATION ##################################################################

# Upload Section
st.title("📊 Sales Forecasting & Risk Analysis Dashboard")
st.sidebar.header("Menu")

st.sidebar.subheader("Upload Your Sales Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xls", "xlsx"])

################################################################## MAIN CONTENT CONFIGURATION ##################################################################
if "df" not in st.session_state:
    st.session_state.df = None

# Show welcome page only if no file is uploaded
if uploaded_file is None:
    st.markdown("""
    ## 👋 Welcome to the Sales Forecasting and Risk Analysis Tool!

    This interactive dashboard helps you:
    - 📊 Analyze data relation and distribution through **Exploratory Data Analysis**.
    - 📈 Forecast future **Revenue** using models like ARIMA, Prophet and Exponential Smoothing.
    - 📉 Analyze **Profitability, Concentration, and Volatility Risks** in your sales data.
    - 🧠 Compare model performance using **Accuracy** scores of models like Random Forest, AdaBoost, Gradient Boosting, and Bagging.
    - 🧾 Visualize **actual vs predicted revenue** trends with interactive plots.

    ### 🧾 What You Need to Do:
    1. Prepare a CSV, XLX or XLSX file containing your sales data.
    2. Make sure it includes columns like `Date`, `Revenue`, `Profit`, `Cost`, and `Units Sold`.
    3. Upload the file using the uploader in the side pannel.
    4. Explore forecasts, model evaluations, and risk profiles.

    ---
    #### Example Column Format:
    - Date: `2023-01-01`
    - Revenue: `10000`
    - Profit: `2500`
    - Cost: `7500`
    - Units Sold: `1200`

    _The dashboard will automatically detect and process your data once the file is uploaded._

    ---
    """)
    st.stop()

# Check if a file has been uploaded
if uploaded_file:
    with st.spinner("Processing your file..."):
        # Process the uploaded file
        loader   = DataLoader()
        df       = loader.load_data(uploaded_file)
        state.df = df
        
        # Notify user of successful upload
        st.success("File processed successfully!")
        st.session_state.df = df
        
        con = Config()
        
        preprocessor = DataInspection(input_data = df)
        
        data_columns = preprocessor.col_names()
        rows_columns = preprocessor.data_shape()
        missing_val  = preprocessor.find_null()
        duplicates   = preprocessor.find_duplicate()
        
        # Count outliers in specified columns
        for col in ['Revenue', 'Units Sold', 'Profit', 'Cost']:
            outliers     = preprocessor.check_outliers(column_name = col, 
                                                       z_thresh    = con.Z_THRESHOLD)
        outliers = len(outliers) if isinstance(outliers, pd.DataFrame) else 0
        
        # Data Preprocessing
        data = preprocessor.change_date_format()
        data = preprocessor.sort_data()
        data = preprocessor.remove_duplicates()
        
        # Handle outliers in specified columns
        for col in ['Revenue', 'Units Sold', 'Profit', 'Cost']:
            data = preprocessor.handle_outliers(window      = con.WINDOW, 
                                                column_name = col)
        processed_data = data

        # Applying Filters
        # Filter Section
        st.sidebar.subheader("🔎 Filter Your Data")

        # Convert dates to Python datetime
        min_date = processed_data["Date"].min().to_pydatetime()
        max_date = processed_data["Date"].max().to_pydatetime()

        # Date Range Filter
        selected_date_range = st.sidebar.slider(
            "Select Date Range:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )

        # Revenue Range Filter
        revenue_min = float(processed_data["Revenue"].min())
        revenue_max = float(processed_data["Revenue"].max())
        selected_revenue = st.sidebar.slider(
            "Revenue Range:",
            min_value=revenue_min,
            max_value=revenue_max,
            value=(revenue_min, revenue_max)
        )

        # Cost Range Filter
        cost_min = float(processed_data["Cost"].min())
        cost_max = float(processed_data["Cost"].max())
        selected_cost = st.sidebar.slider(
            "Cost Range:",
            min_value=cost_min,
            max_value=cost_max,
            value=(cost_min, cost_max)
        )

        # Profit Range Filter
        profit_min = float(processed_data["Profit"].min())
        profit_max = float(processed_data["Profit"].max())
        selected_profit = st.sidebar.slider(
            "Profit Range:",
            min_value=profit_min,
            max_value=profit_max,
            value=(profit_min, profit_max)
        )

        # Units Sold Range Filter
        units_min = float(processed_data["Units Sold"].min())
        units_max = float(processed_data["Units Sold"].max())
        selected_units = st.sidebar.slider(
            "Units Sold Range:",
            min_value=units_min,
            max_value=units_max,
            value=(units_min, units_max)
        )

        filtered_data = processed_data[
                                (processed_data["Date"] >= selected_date_range[0]) &
                                (processed_data["Date"] <= selected_date_range[1]) &
                                (processed_data["Revenue"].between(selected_revenue[0], selected_revenue[1])) &
                                (processed_data["Cost"].between(selected_cost[0], selected_cost[1])) &
                                (processed_data["Profit"].between(selected_profit[0], selected_profit[1])) &
                                (processed_data["Units Sold"].between(selected_units[0], selected_units[1]))
                            ]


        #Exploratory Data Analysis
        eda = ExploratoryDataAnalysis(input_data = filtered_data)
        
        description   = preprocessor.data_description()
        revenue_sum   = eda.total_revenue()
        revenue_avg   = eda.average_revenue()
        profit_margin = eda.profit_margin()
        total_units   = eda.total_units()
        
        # Performing COVID Analysis
        ca              = CovidAnalysis(input_data = filtered_data)
        covid, precovid = ca.set_covid_dates()
        
        # Forecasting
        forecast     = Forecasting(filtered_data)
        arima_plot   = forecast.train_arima(future_steps = con.FUTURE_STEPS)
        prophet_plot = forecast.train_prophet(future_steps = con.FUTURE_STEPS)
        exp_plot     = forecast.train_exponential_smoothing(future_steps = con.FUTURE_STEPS)

        # Display Data Overview
        st.subheader("Data Overview")
        with st.expander("Click to view data sample", expanded = True):
            st.dataframe(processed_data.head(10), 
                         use_container_width = True)
        
        # Display Data Summary 
        st.subheader("General Data Summary")
        with st.expander("Click to view data summary"):
            col1, col2, col3, col4, col5 = st.columns(spec = 5)
            
            with col1:
                st.markdown("**<small>No. of Rows:</small>**", 
                            unsafe_allow_html = True)
                
                st.markdown(f"<h3 style = text-align: center;>{rows_columns[0]}</h3>", 
                            unsafe_allow_html = True)
                
            
            with col2:
                st.markdown("**<small>No. of Columns:</small>**",
                            unsafe_allow_html = True)
                
                st.markdown(f"<h3 style = text-align: center;>{rows_columns[1]}</h3>",
                            unsafe_allow_html = True)
                
            with col3:
                st.markdown("**<small>Missing Values:</small>**", 
                            unsafe_allow_html = True)
                
                st.markdown(f"<h3 style = text-align: center;>{missing_val}</h3>", 
                            unsafe_allow_html = True)
                
            with col4:
                st.markdown("**<small>Duplicates:</small>**", 
                            unsafe_allow_html = True)
                
                st.markdown(f"<h3 style = text-align: center;>{duplicates}</h3>", 
                            unsafe_allow_html = True)
                
            with col5:
                st.markdown("**<small>Outliers:</small>**", 
                            unsafe_allow_html = True)
                
                st.markdown(f"<h3 style = text-align: center;>{outliers}</h3>", 
                            unsafe_allow_html = True)

        # Display KPI Metrics
        st.subheader("KPI Metrics")
        with st.expander("Click to view KPI Metrics", expanded = True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Revenue Sum", revenue_sum)
            col2.metric("Revenue Avg", revenue_avg)
            col3.metric("Profit Margin", profit_margin)
            col4.metric("Total Units", total_units)

        # Statistical Summary 
        st.subheader("Statistical Summary")
        with st.expander("Click to view statistical summary"):
            st.dataframe(description, use_container_width = True)

        # EDA Plot Visualization
        st.subheader("Exploratory Data Analysis Visualizations")
        eda_plot_options = ['Bar Chart on Country', 
                            'Bar Chart on Product', 
                            'Pie Chart', 
                            'Grouped Bar Chart', 
                            'Sales Trend', 
                            'Bubble Plot', 
                            'Heatmap', 
                            'Decomposition']
        selected_plot = st.sidebar.selectbox("Select a plot", eda_plot_options)

        with st.expander("Click to view the plots"):

            if selected_plot == 'Bar Chart on Country':
                st.plotly_chart(eda.barplot_total_sale_by_column('Country'), use_container_width = True)
                
            elif selected_plot == "Bar Chart on Product":
                st.plotly_chart(eda.barplot_total_sale_by_column('Product'), use_container_width = True)
                
            elif selected_plot == "Pie Chart":
                st.plotly_chart(eda.pieplot_for_product_sale(), use_container_width = True)
                
            elif selected_plot == "Grouped Bar Chart":
                st.plotly_chart(eda.groupedbar_product_over_country(), use_container_width = True)
                
            elif selected_plot == "Sales Trend":
                st.plotly_chart(eda.sales_trend_per_product(), use_container_width = True)
                
            elif selected_plot == "Bubble Plot":
                st.plotly_chart(eda.simple_bubble_plot(), use_container_width = True)
                
            elif selected_plot == "Heatmap":
                st.plotly_chart(eda.heatmap_for_product_profit(), use_container_width = True)
                
            elif selected_plot == "Decomposition":
                st.plotly_chart(eda.decompose_revenue(), use_container_width = True)

        # Covid Analysis
        st.subheader("COVID Analysis")
        with st.expander("Click to view COVID Analysis"):
            col1, col2, col3 = st.columns(3)
            
        with col1:
            st.markdown("**<small>Average Revenue Pre-COVID:</small>**", 
                        unsafe_allow_html = True)
            st.markdown(f"<h3 style = text-align: center;>{ca.average_revenue(precovid)}</h3>", 
                        unsafe_allow_html = True)
            st.markdown("**<small>Average Revenue During COVID:</small>**", 
                        unsafe_allow_html = True)
            st.markdown(f"<h3 style = text-align: center;>{ca.average_revenue(covid)}</h3>", 
                        unsafe_allow_html = True)
            
        with col2:
            st.markdown("**<small>Average Units Sold Pre-COVID:</small>**", 
                        unsafe_allow_html = True)
            st.markdown(f"<h3 style = text-align: center;>{ca.average_units_sold(precovid)}</h3>", 
                        unsafe_allow_html = True)
            st.markdown("**<small>Average Units Sold During COVID:</small>**", 
                        unsafe_allow_html = True)
            st.markdown(f"<h3 style = text-align: center;>{ca.average_units_sold(covid)}</h3>", 
                        unsafe_allow_html = True)
            
        with col3:
            st.markdown("**<small>Profit Margin Pre-COVID:</small>**", 
                        unsafe_allow_html = True)
            st.markdown(f"<h3 style = text-align: center;>{ca.profit_margin(precovid)}</h3>", 
                        unsafe_allow_html = True)
            st.markdown("**<small>Profit Margin During COVID:</small>**", 
                        unsafe_allow_html = True)
            st.markdown(f"<h3 style = text-align: center;>{ca.profit_margin(covid)}</h3>", 
                        unsafe_allow_html = True)
            
        st.subheader("Average Sales Trend")
        with st.expander("Click to view Average Sales Trend During COVID"):
            covid_plots = ['Covid Sale Product', 'Covid Sale Country']
            selected_covid_plot = st.sidebar.selectbox("Select a COVID Sale Trend Plot", covid_plots)
            if selected_covid_plot == 'Covid Sale Product':
                st.plotly_chart(ca.average_sales_trend(covid, 'Product'), 
                                use_container_width = True)
                
            elif selected_covid_plot == 'Covid Sale Country':
                st.plotly_chart(ca.average_sales_trend(covid, 'Country'), 
                                use_container_width = True)
        
        with st.expander("Click to view Average Sales Trend Pre-COVID"):
            precovid_plots = ['Precovid Sale Product', 'Precovid Sale Country']
            selected_precovid_plot = st.sidebar.selectbox("Select a Pre-COVID Sale Trend Plot", precovid_plots)
            if selected_precovid_plot == 'Precovid Sale Product':
                st.plotly_chart(ca.average_sales_trend(precovid, 'Product'), 
                                use_container_width = True)
            elif selected_precovid_plot == 'Precovid Sale Country':
                st.plotly_chart(ca.average_sales_trend(precovid, 'Country'), 
                                use_container_width = True)

        # Risk Analysis
        ra = RiskAnalysis(input_data = processed_data)
        st.subheader("Risk Analysis")
        with st.expander("Click to view Concentration Risk Analysis"):
            risk_plots = ['Concentration on Product', 'Concentration on Country', 'Profitability Risk', 'Volatility Risk']
            selected_risk = st.sidebar.selectbox("Select a Risk Analysis", risk_plots)

            if selected_risk == 'Concentration on Product':
                st.plotly_chart(ra.calculate_concentration_risk('Product'), 
                                use_container_width = True)
            elif selected_risk == 'Concentration on Country':
                st.plotly_chart(ra.calculate_concentration_risk('Country'), 
                                use_container_width = True)
        
            elif selected_risk == 'Profitability Risk':
                st.plotly_chart(ra.calculate_profitability_risk(), 
                                use_container_width = True)
            
            elif selected_risk == 'Volatility Risk':
                st.plotly_chart(ra.calculate_volatility_risk(value_cols = ['Revenue', 'Profit']), 
                                use_container_width = True)

        # Forecasting Analysis
        st.subheader("Forecasting")
    
        if "section" not in st.session_state:
            st.session_state.section = "ARIMA Forecast"
            
        cols = st.columns([1,1,1], gap = "small")
            
        if cols[0].button("ARIMA Forecast"):
                st.session_state.section = "ARIMA Forecast"
                
        if cols[1].button("Prophet Forecast"):
                st.session_state.section = "Prophet Forecast"
                
        if cols[2].button("Exponential Smoothing Forecast"):
                st.session_state.section = "Exponential Smoothing Forecast"
        with st.expander("Click to view Forecast"):       
            if st.session_state.section == "ARIMA Forecast":
                st.plotly_chart(arima_plot, use_container_width = True)
                
            elif st.session_state.section == "Prophet Forecast":
                st.plotly_chart(prophet_plot, use_container_width = True)
                
            elif st.session_state.section == "Exponential Smoothing Forecast":
                st.plotly_chart(exp_plot, use_container_width = True)


        # Model Fitting
        def train_models(data: pd.DataFrame, target_column: str, models: list, n_splits: int = 5):
            """Main function for Streamlit or server integration"""
            fitter = ModelFitter(df=data, target_col=target_column, selected_models=models)
            fitter.fit(n_splits=n_splits)
            results_df = fitter.get_results_df()
            accuracy_plot = fitter.get_evaluation_plot()
            actual_predicted_plots = {
                model: fitter.get_actual_vs_predicted_plot(model)
                for model in models
            }
            return results_df, accuracy_plot, actual_predicted_plots


    
        # Call the function
        results_df, accuracy_plot, actual_vs_predicted = train_models(
            data=df,
            target_column="Revenue",
            models=["RandomForest", "GradientBoosting", "AdaBoost", "Bagging"],
            n_splits=5
        )

        # Show bar chart
        st.subheader("Accuracy of Model Fitted")
        with st.expander("Click to view model accuracy"):
            st.plotly_chart(accuracy_plot)
            # Show metrics table
            st.dataframe(results_df, use_container_width = True)

        with st.expander("Click to see model actual vs predicted plot"):
            model_names = ['AdaBoost', 'RandomForest', 'Bagging', 'GradientBoosting']
            selected_model = st.sidebar.selectbox("Select Model", model_names, key="model_select_plot")

            st.subheader(f"Actual vs Predicted - {selected_model}")

            fig = actual_vs_predicted.get(selected_model)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No predictions available for {selected_model}")

        st.subheader("Executive Summary")
        with st.expander("Click to view data analysis summary"):
            st.markdown("""
                ### Executive Summary
                **Project Title:** Interactive Sales Forecasting and Risk Analysis Dashboard

                #### Objective:
                To create a comprehensive dashboard that enables business stakeholders to upload sales data, explore sales trends, assess business risks, and forecast future revenue using machine learning and time series models.

                #### Project Workflow
                1. **Data Loading**
                - Users can upload datasets in .csv, .xls, or .xlsx format.

                2. **Data Preprocessing**
                - No missing values detected.
                - 525 duplicate rows were removed (reduced data from 1225 to 700 rows).
                - 12 outliers were identified and handled.
                - Dates were converted from Excel's serial format to standard YYYY-MM-DD.
                - Dataset was sorted chronologically for time-based analysis.

                3. **Exploratory Data Analysis (EDA)**
                ***Descriptive Metrics:***
                - Total revenue, average revenue, total units sold, and overall profit margin.

                ***Visual Insights:***
                - Bar plots for revenue by country and by product.
                - Pie chart showing product sales distribution.
                - Grouped bar plot showing units sold per product across countries.
                - Line chart for product-wise revenue trends over time.
                - Bubble plot comparing profit vs. revenue based on units sold.
                - Heatmap of profit distribution across countries and products.
                - Seasonal decomposition of monthly revenue trends.

                4. **COVID-19 Impact Analysis**      
                - Data was split into COVID-period and non-COVID period.
                - Compared average revenue, units sold, and profit margin across both periods.
                - Plotted average sale trends per product and country during and outside COVID for deeper insights.

                5. **Risk Analysis**
                - ***Concentration Risk:*** Assessed through product and country dominance using pie charts.
                - ***Volatility Risk:*** Examined based on revenue and profit fluctuations.
                - ***Profitability Risk:*** Evaluated using profit margin variability.

                6. **Sales Forecasting**
                - ***Time Series models used:***
                        
                    a. ARIMA (performed best), Prophet, and Exponential Smoothing.
                        
                    b. Models compared for forecast reliability and visualized with future revenue trends.

                7. **Machine Learning Model Fitting**
                - ***Regression models used:***
                        
                    a. Random Forest, Gradient Boosting, AdaBoost, Bagging.
                        
                    b. Random Forest delivered the highest prediction accuracy.
                        
                    c. Accuracy of each model shown in a bar chart and detailed in a results table.
                        
                    d. Interactive plots of Actual vs. Predicted values for each model enhance interpretability.

                #### Business Value
                - Enables strategic decision-making by identifying high-performing products and markets.
                - Provides actionable insights on how external events (like COVID-19) affect sales.
                - Forecasts help with resource allocation, inventory planning, and revenue expectations.
                - Risk analysis equips businesses with tools to minimize financial uncertainty.
            """)