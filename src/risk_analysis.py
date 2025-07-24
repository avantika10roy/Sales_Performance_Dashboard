from config import Config
import plotly.express as px
import plotly.graph_objects as go

con = Config()

class RiskAnalysis:
    def __init__(self, input_data):
        self.input_data = input_data

    def calculate_concentration_risk(self, column):
        column_revenue = self.input_data.groupby(column, as_index = False)['Revenue'].sum()
        # column_revenue_percentage = (column_revenue/column_revenue.sum()) * 100

        piechart = px.pie(column_revenue,
                          values                  = 'Revenue',
                          names                   = column,
                          title                   = (f'Percentage of Revenue generated from each {column}: Concentration Risk Analysis'),
                          color_discrete_sequence = con.COLOR_PALETTE,
                          height                  = con.HEIGHT,
                          width                   = con.WEIGHT)
        piechart.update_layout(autosize = False)
        return piechart
    
    def calculate_profitability_risk(self):
        self.input_data["Profit Margin"] = self.input_data['Profit'] / self.input_data['Revenue']
    
        fig = px.histogram(self.input_data, 
                           x                       = "Profit Margin", 
                           nbins                   = 30, 
                           title                   = "Profit Margin Distribution", 
                           marginal                = "box",  # Optional: adds mini boxplot on side
                           color_discrete_sequence = ["green"])
        
        fig.update_layout(
            xaxis_title="Profit Margin",
            yaxis_title="Count",
            template="plotly_white"
        )

        return fig

    def calculate_volatility_risk(self, value_cols=['Revenue', 'Profit']):
        self.input_data["Month"] = self.input_data['Date'].dt.to_period("M")
        monthly = self.input_data.groupby("Month")[value_cols].sum()
        monthly.index = monthly.index.to_timestamp()

        fig = go.Figure()
        for col in value_cols:
            fig.add_trace(go.Scatter(
                                    x    = monthly.index,
                                    y    = monthly[col],
                                    mode = "lines+markers",
                                    name = col
                                ))

        fig.update_layout(
                            title       = "Monthly Revenue and Profit Volatility",
                            xaxis_title = "Month",
                            yaxis_title = "Amount",
                            template    = "plotly_white"
                        )
        
        return fig
