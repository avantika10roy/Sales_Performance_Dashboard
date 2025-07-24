# Importing Necessary Dependencies
import pandas as pd
from config import Config
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

con = Config()

class ExploratoryDataAnalysis:
    def __init__(self, input_data):
            self.input_data = input_data

    def total_revenue(self):
        return self.input_data['Revenue'].sum()

    def average_revenue(self):
        return round(self.input_data['Revenue'].mean(), 2)

    def profit_margin(self):
        total_profit  = self.input_data['Profit'].sum()
        total_revenue = self.input_data['Revenue'].sum()
        profit_margin = (total_profit / total_revenue) * 100
        return round(profit_margin, 2)

    def total_units(self):
        total_units_sold = self.input_data['Units Sold'].sum()
        return total_units_sold

    def top_3_countries(self):
        grouped = self.input_data.groupby("Country").agg(
            Total_Revenue    = pd.NamedAgg(column  = "Revenue", 
                                           aggfunc = "sum"),
            Total_Profit     = pd.NamedAgg(column  = "Profit", 
                                           aggfunc = "sum"),
            Total_Units_Sold = pd.NamedAgg(column  = "Units Sold", 
                                           aggfunc = "sum")
        ).reset_index()

        # Calculate profit percentage
        grouped["Profit_Percentage"] = (grouped["Total_Profit"] / grouped["Total_Revenue"]) * 100

        # Round for display
        grouped["Profit_Percentage"] = grouped["Profit_Percentage"].round(2)

        # Get top 3 companies by revenue
        top_3 = grouped.sort_values(by        = "Total_Revenue", 
                                    ascending = False).head(3)
        return top_3
    
    def top_3_products(self):
        grouped = self.input_data.groupby("Product").agg(
            total_revenue     = pd.NamedAgg(column  = "Revenue",
                                            aggfunc = "sum"),
            total_profit      = pd.NamedAgg(column  = "Profit",
                                            aggfunc = "sum"),
            total_units_sold  = pd.NamedAgg(column  = "Units Sold",
                                            aggfunc = "sum")
        ).reset_index()

        grouped['profit_percent'] = (grouped['total_profit'] / grouped['total_revenue']) * 100
        grouped['profit_percent'] = grouped['profit_percent'].round(2)

        top_3 = grouped.sort_values(by        = "total_revenue",
                                    ascending = False).head(3)
        return top_3

    def barplot_total_sale_by_column(self, column_name: str = None):
        barplot = px.bar(self.input_data, 
                    x                       = column_name, 
                    y                       = "Revenue",
                    title                   = (f"Total Revenue by {column_name}"),
                    hover_name              = 'Product',
                    hover_data              = ['Units Sold', 'Cost', 'Profit'],
                    color                   = column_name,  
                    color_discrete_sequence = con.COLOR_PALETTE,
                    height                  = con.HEIGHT,
                    width                   = con.WEIGHT)
        
        return barplot

    def pieplot_for_product_sale(self):
        # Group by products and sum the units sold
        product_units = self.input_data.groupby("Product", as_index = False)["Units Sold"].sum()
        # Plot pie chart
        piechart = px.pie(product_units, 
                            values                  = 'Units Sold', 
                            names                   = 'Product',
                            title                   = 'Percentage of Products Sold',
                            color_discrete_sequence = con.COLOR_PALETTE,
                            height                  = con.HEIGHT,
                            width                   = con.WEIGHT
                            )
        piechart.update_layout(autosize = False)
        return piechart

    def groupedbar_product_over_country(self):
        # Plot stacked bar chart
        grouped_bar = px.bar(self.input_data,
                             x                       = 'Country',
                             y                       = 'Units Sold',
                             color                   = 'Product',
                             barmode                 = 'group',
                             title                   = 'Units Sold per Product in Each Country',
                             hover_data              = ['Cost', 'Profit', 'Date'],
                             hover_name              = 'Product',
                             color_discrete_sequence = con.COLOR_PALETTE,
                             height                  = con.HEIGHT,
                             width                   = con.WEIGHT)
        return grouped_bar

    def sales_trend_per_product(self):
        sales_trend = self.input_data.groupby(["Date", "Product"])["Revenue"].sum().reset_index()
        # Plot
        linechart = px.line(sales_trend,
                            x                       = 'Date',
                            y                       = 'Revenue',
                            color                   = 'Product',
                            hover_name              = 'Product',
                            title                   = 'Sales Trend per Product Over Time',
                            height                  = con.HEIGHT,
                            width                   = con.WEIGHT,
                            text                    = 'Date',
                            markers                 = True,
                            color_discrete_sequence = con.COLOR_PALETTE,
                            symbol                  = 'Product')
        linechart.update_traces(mode='lines+markers')
        return linechart


    def simple_bubble_plot(self):
        grouped = self.input_data.groupby("Product").agg({
                                    "Revenue": "sum",
                                    "Profit": "sum",
                                    "Units Sold": "sum"
                                }).reset_index()
                                
        bubble_plot = px.scatter(grouped,
                                 x = grouped['Revenue'],
                                 y = grouped['Profit'],
                                 color = 'Product',
                                 size = grouped['Units Sold'],
                                 title = 'Bubble Plot: Revenue vs Profit (Size = Units Sold)',
                                 color_discrete_sequence = con.COLOR_PALETTE
                                 )
        return bubble_plot

    def heatmap_for_product_profit(self):
        # Heatmap: Profit by Product and Country
        pivot = self.input_data.pivot_table(values  = "Profit", 
                                       index   = "Country", 
                                       columns = "Product", 
                                       aggfunc = "sum")
        heatmap = px.imshow(pivot,
                            text_auto = True,
                            aspect  = 'auto',
                            title = 'Profit by Country and Product',
                            color_continuous_scale=px.colors.sequential.Viridis)

        return heatmap

    def decompose_revenue(self):
        self.input_data = self.input_data.groupby('Date')['Revenue'].sum().reset_index()

        # Perform seasonal decomposition
        result = seasonal_decompose(self.input_data['Revenue'], model='additive', period=3)

        # Create a Plotly subplot with 4 rows (original, trend, seasonal, residual)
        decomposition_plot = make_subplots(rows = 4, 
                                           cols = 1, 
                                           shared_xaxes = True,
                                           subplot_titles = ("Original Series", 
                                                             "Trend", 
                                                             "Seasonal", 
                                                             "Residual"))

        # Original series
        decomposition_plot.add_trace(go.Scatter(x    = self.input_data.index, 
                                                y    = self.input_data['Revenue'],
                                                mode = 'lines', 
                                                name = 'Original',
                                                line=dict(color='red')),
                                     row = 1, 
                                     col = 1)

        # Trend
        decomposition_plot.add_trace(go.Scatter(x    = self.input_data.index, 
                                                y    = result.trend,
                                                mode = 'lines', 
                                                name = 'Trend',
                                                line=dict(color='blue')),
                                     row = 2, 
                                     col = 1)

        # Seasonal
        decomposition_plot.add_trace(go.Scatter(x    = self.input_data.index, 
                                                y    = result.seasonal,
                                                mode = 'lines', 
                                                name = 'Seasonal',
                                                line=dict(color='green')),
                                     row = 3, 
                                     col = 1)

        # Residual
        decomposition_plot.add_trace(go.Scatter(x = self.input_data.index, 
                                                y = result.resid,
                                                mode = 'markers', 
                                                name = 'Residual',
                                                marker=dict(color='red')),
                                     row = 4, 
                                     col = 1)

        # Layout settings
        decomposition_plot.update_layout(height = con.HEIGHT, 
                                         width  = con.WEIGHT,
                                         title_text = "Additive Decomposition of Monthly Revenue (Period = 3)",
                                         showlegend = False)

        return decomposition_plot

