"""
    implementation of logistic regression
"""

import numpy as np

class LogisticRegression:
    def __init__(self, fit_intercept=True, max_iter=100, tol=1e-4):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.max_iter = max_iter
        self.tol = tol
        
    def train(self, X, y):
        pass    
    
    
    def predict_proba(self, X):
        pass

    
    def predict(self, X, threshold=0.5):
        pass
