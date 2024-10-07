import pandas as pd
import json
class REDataset():
    def __init__(self, path) -> None:
        self.dataset_path = path
        self.df = pd.read_csv(self.dataset_path)

    def load_train_data(self):
        return self.df.entities.apply(json.loads), self.df.relations.apply(json.loads)
    
    def load_test_data(self):
        return self.df.entities.apply(json.loads)
    
        