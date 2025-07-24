import os
import pandas as pd

class DataLoader:
    def __init__(self):
        pass

    def load_data(self, input_data: str = None):
        try:
            ext = os.path.splitext(input_data.name)[1].lower()
            if ext == '.csv':
                data = pd.read_csv(filepath_or_buffer = input_data)
            elif ext == ['xlx', 'xlsx']:
                data = pd.read_excel(filepath_or_buffer = input_data)
            else:
                print("Please upload file with extensions: .csv, .xlx, .xlsx")

            return data

        except FileNotFoundError as e:
            raise ValueError(f"Could not find the file you were looking for : {e}")