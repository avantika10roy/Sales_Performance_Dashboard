import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class Forecasting:
    def __init__(self, input_data):
        self.input_data = input_data
        self.last_date = self.input_data['Date'].iloc[-1]

    def train_arima(self, future_steps):
        arima_model    = ARIMA(self.input_data['Revenue'], order = (5,1,0))
        arima_result   = arima_model.fit()
        arima_forecast = arima_result.forecast(steps = future_steps)
        future_dates   = pd.date_range(start   = self.last_date + pd.Timedelta(days=1), 
                                       periods = future_steps)

        fig_arima = go.Figure()
        fig_arima.add_trace(go.Scatter(x    = future_dates, 
                                       y    = arima_forecast, 
                                       name = "ARIMA Forecast", 
                                       line = dict(dash  = 'dash', 
                                                   color = 'blue')))
        fig_arima.update_layout(
                            title       = "ARIMA Revenue Forecast (Next 90 Days)",
                            xaxis_title = "Date",
                            yaxis_title = "Revenue",
                            template    = "plotly_white",
                            hovermode   = "x unified"
                        )
        return fig_arima
    
    def train_prophet(self, future_steps):
        prophet_df = self.input_data.rename(columns = {"Date"    : "ds", 
                                                       "Revenue" : "y"})
        prophet_model = Prophet()
        prophet_model.fit(prophet_df)
        future           = prophet_model.make_future_dataframe(periods = future_steps)
        forecast         = prophet_model.predict(future)
        prophet_forecast = forecast[['ds', 'yhat']].tail(future_steps)
            
        fig_prophet = go.Figure()
        fig_prophet.add_trace(go.Scatter(x    = prophet_forecast['ds'], 
                                         y    = prophet_forecast['yhat'], 
                                         name = "Prophet Forecast", 
                                         line = dict(dash  = 'dot', 
                                                     color = 'green')))
        fig_prophet.update_layout(
                            title="Prophet Revenue Forecast (Next 90 Days)",
                            xaxis_title = "Date",
                            yaxis_title = "Revenue",
                            template    = "plotly_white",
                            hovermode   = "x unified"
                        )
        return fig_prophet
    
    def train_exponential_smoothing(self, future_steps):
        ets_model    = ExponentialSmoothing(self.input_data['Revenue'], 
                                            trend                 = 'add', 
                                            seasonal              = None, 
                                            initialization_method = 'estimated')
        ets_result   = ets_model.fit()
        ets_forecast = ets_result.forecast(steps = future_steps)
        future_dates = pd.date_range(start   = self.last_date + pd.Timedelta(days=1), 
                                     periods = future_steps)
    
        fig_ets = go.Figure()
        fig_ets.add_trace(go.Scatter(x    = future_dates, 
                                     y    = ets_forecast, 
                                     name = "ETS Forecast", 
                                     line = dict(dash  = 'longdash', 
                                                 color = 'red')))
        fig_ets.update_layout(
                            title       = "ETS Revenue Forecast (Next 90 Days)",
                            xaxis_title = "Date",
                            yaxis_title = "Revenue",
                            template    = "plotly_white",
                            hovermode   = "x unified"
                        )
        return fig_ets
