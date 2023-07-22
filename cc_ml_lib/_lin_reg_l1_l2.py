"""
    linear regression with both l1 and l2
"""

import numpy as np

class LinearRegressionWithL1L2:
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.max_iter = max_iter
        
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass
