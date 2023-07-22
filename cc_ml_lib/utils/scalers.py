import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.stds = X.std(axis=0)
        print('Means\n', self.mean)
        print('STD\n', self.stds)
            
    def transform(self, X):
        X = (X - self.mean) / self.stds
        return X