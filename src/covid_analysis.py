from config import Config
import plotly.express as px

con = Config()

class CovidAnalysis:
    def __init__(self, input_data):
        self.input_data = input_data

    def set_covid_dates(self):
        covid = self.input_data[(self.input_data['Date'] >= '2020-02-01') & \
                       (self.input_data['Date'] <= '2020-12-01')]
        
        precovid    = self.input_data[(self.input_data['Date'] >= '2019-09-01') & \
                       (self.input_data['Date'] <'2020-02-01')]
        
        return covid, precovid
    
    def average_revenue(self, data):
        average_revenue = round(data['Revenue'].mean(), 2)
        return average_revenue
    
    def average_units_sold(self, data):
        average_units = round(data['Units Sold'].mean(), 2)
        return average_units
    
    def profit_margin(self, data):
        total_profit  = data['Profit'].sum()
        total_revenue = data['Revenue'].sum()
        profit_margin = (total_profit / total_revenue) * 100
        return round(profit_margin, 2)

    def average_sales_trend(self, data, column):
        average_revenue_trend = round(data.groupby(['Date', column])['Revenue'].mean(), 2).reset_index()
        line_chart = px.line(average_revenue_trend,
                             x                       = 'Date',
                             y                       = 'Revenue',
                             color                   = column,
                             hover_name              = column,
                             title                   = (f"Average Revenue Trend per {column} over Time"),
                             height                  = con.HEIGHT,
                             width                   = con.WEIGHT,
                            #  text                    = 'Date',
                             markers                 = True,
                             color_discrete_sequence = con.COLOR_PALETTE,
                             symbol                  = column)
        # line_chart.update_traces(mode = 'line+markers')
        return line_chart
    
    
    

