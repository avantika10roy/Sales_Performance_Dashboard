# Importing Necessary Dependencies
import numpy as np
import pandas as pd
from config import Config

con = Config() 

class DataInspection:
    def __init__(self, input_data: str = None):
        try:
            self.input_data  = input_data
        except FileNotFoundError as e:
            print(f"No file found: {e}")

    def data_description(self):
        return self.input_data.describe()

    def data_info(self):
        return self.input_data.info()

    def col_names(self):
        return self.input_data.columns

    def data_shape(self):
        return self.input_data.shape

    def find_null(self):
        return self.input_data.isnull().sum().sum()
        

    def change_date_format(self):
        self.input_data['Date'] = (pd.to_datetime(self.input_data['Date'], 
                                                 origin  = con.EXCEL_START_DATE, 
                                                 unit    = 'D') - pd.Timedelta(days = 2)).dt.strftime("%Y-%m-%d")
        return self.input_data

    def sort_data(self):
        # Convert 'Date' column to datetime
        self.input_data['Date'] = pd.to_datetime(self.input_data['Date'], format="%Y-%m-%d")
    
        # Sort the entire DataFrame by date
        self.input_data = self.input_data.sort_values(by='Date').reset_index(drop=True)
    
        return self.input_data

    def find_duplicate(self):
        return self.input_data.duplicated().sum()

    def remove_duplicates(self):
        self.input_data =  self.input_data.drop_duplicates()
        return self.input_data

    def check_outliers(self, column_name: str = None, z_thresh = con.Z_THRESHOLD):
        try:
            z_scores = (self.input_data[column_name] - self.input_data[column_name].mean()) / self.input_data[column_name].std()
            outliers = self.input_data[np.abs(z_scores) > z_thresh]
            return outliers

        except Exception as e:
            print(f"Unable to detect outliers: {e}")

    def handle_outliers(self, window: int = con.WINDOW, column_name: str = None):
        if column_name is None:
            raise ValueError("Please provide a column name.")

        # Work on a copy of the column to avoid SettingWithCopyWarning
        series = self.input_data[column_name].copy()

        # Calculate IQR bounds
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Detect Outliers
        outlier = (series < lower) | (series > upper)

        # Replace each outlier
        for i in range(len(series)):
            if outlier.iloc[i]:
                start = max(0, i - window)
                end = min(len(series), i + window + 1)

                # Get surrounding values and mask
                surrounding = series.iloc[start:end]
                surrounding_mask = outlier.iloc[start:end]
                surrounding = surrounding[~surrounding_mask]

                if not surrounding.empty:
                    series.iloc[i] = int(round(surrounding.mean()))
                else:
                    # Fallback: use global median if all surrounding values are also outliers
                    series.iloc[i] = int(round(series[~outlier].median()))

        # Update only the target column in the original DataFrame
        self.input_data[column_name] = series

        return self.input_data  # Optional: return updated DataFrame