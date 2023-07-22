"""
    linear regression with l2
"""

import numpy as np

class LinearRegressionWithL2:
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X, y):
        pass

    
    def predict(self, X):
        pass
