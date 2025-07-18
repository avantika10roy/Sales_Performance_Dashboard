import pandas as pd

class DataLoader:
    def __init__(self):
        pass

    def load_data(self, input_data: str = None):
        try:
            data = pd.read_csv(filepath_or_buffer = input_data)
            return data

        except FileNotFoundError as e:
            raise ValueError(f"Could not find the file you were looking for : {e}")