import pandas as pd
import os

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.file_type = self._get_file_type()

    def _get_file_type(self):
        _, ext = os.path.splitext(self.file_path)
        return ext.lower()

    def load_data(self):
        if self.file_type == '.csv':
            self.data = pd.read_csv(self.file_path)
            self.data =  self.data.iloc[:100]
        elif self.file_type == '.json':
            self.data = pd.read_json(self.file_path)
        elif self.file_type == '.xls':
            self.data = pd.read_excel(self.file_path)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")
        
        return self.data

    def get_preview(self, rows=5):
        if self.data is None:
            raise ValueError("Data not loaded.")
        
        if isinstance(self.data, pd.DataFrame):
            return self.data.head(rows)
        else:
            raise ValueError("Something went wrong")
    
    def get_shape(self):
        if self.data is None:
            raise ValueError("Data not loaded.")
        if isinstance(self.data, pd.DataFrame):
            return self.data.shape
            