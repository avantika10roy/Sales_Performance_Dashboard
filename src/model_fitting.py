import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor
)

class ModelFitter:
    def __init__(self, df, target_col, selected_models=None):
        self.df = df.copy()
        self.target_col = target_col
        self.available_models = {
            "RandomForest": RandomForestRegressor(),
            "AdaBoost": AdaBoostRegressor(),
            "Bagging": BaggingRegressor(),
            "GradientBoosting": GradientBoostingRegressor()
        }
        self.models = {name: self.available_models[name] for name in selected_models} if selected_models else self.available_models
        self.results = {}
        self.last_split_predictions = {}  # store for actual vs predicted plot
        self.X, self.y = self._prepare_data()

    def _prepare_data(self):
        df = self.df.copy()
        df.drop_duplicates(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # One-hot encode categorical variables
        for col in ['Country', 'Product']:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], drop_first=True)

        self.df = df  # updated copy with encoded columns
        X = df.drop(columns=['Date', self.target_col])
        y = df[self.target_col]
        return X, y

    def fit(self, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for name, model in self.models.items():
            acc_list = []

            for split_index, (train_idx, test_idx) in enumerate(tscv.split(self.X)):
                X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mape = mean_absolute_percentage_error(y_test, y_pred)
                accuracy = 1 - mape
                acc_list.append(accuracy)

                # Save last split predictions for visualization
                if split_index == (n_splits - 1):
                    self.last_split_predictions[name] = {
                        'date': self.df['Date'].iloc[test_idx].values,
                        'actual': y_test.values,
                        'predicted': y_pred
                    }

            self.results[name] = {
                'avg_accuracy': round(np.mean(acc_list) * 100, 2)  # percentage
            }

    def get_results_df(self):
        return pd.DataFrame(self.results).T.reset_index().rename(columns={'index': 'Model'})

    def get_evaluation_plot(self):
        df = self.get_results_df()
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Accuracy (%)', x=df['Model'], y=df['avg_accuracy'], marker_color='mediumseagreen'))
        fig.update_layout(
            title='Model Accuracy Comparison (Higher is Better)',
            xaxis_title='Model',
            yaxis_title='Accuracy (%)',
            yaxis_range=[0, 100],
            template='plotly_white'
        )
        return fig

    def get_actual_vs_predicted_plot(self, model_name):
        if model_name not in self.last_split_predictions:
            return None

        data = self.last_split_predictions[model_name]

        # Convert to DataFrame for grouping
        df_plot = pd.DataFrame({
            'Date': pd.to_datetime(data['date']),
            'Actual': data['actual'],
            'Predicted': data['predicted']
        })

        # Group by Date (daily average); use 'W' for weekly if needed
        df_grouped = df_plot.groupby('Date', as_index=False).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_grouped['Date'], y=df_grouped['Actual'],
                                mode='lines+markers', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df_grouped['Date'], y=df_grouped['Predicted'],
                                mode='lines+markers', name='Predicted', line=dict(color='orange')))
        fig.update_layout(
            title=f"Actual vs Predicted Revenue ({model_name}) - Last Split",
            xaxis_title='Date',
            yaxis_title='Revenue',
            template='plotly_white',
            legend=dict(x=0, y=1.1, orientation="h")
        )
        return fig

