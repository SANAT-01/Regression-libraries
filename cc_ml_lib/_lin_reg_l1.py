"""
    linear regression with l1
"""

import numpy as np

class LinearRegressionWithL1:
    
    def __init__(self, alpha=0.01, max_iter=1000, tol=0.01):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None 
    
    def fit(self, X, y):
        n, p = X.shape
        self.coef_ = np.random.rand(p)
        self.intercept_ = np.mean(y)
        prev = float('inf')
        for iteration in range(self.max_iter):
            beta_old = np.copy(self.coef_)
            for j in range(p):
                X_j = X[:, j]
                y_pred = self.predict(X) + self.intercept_ - self.coef_[j]*X_j
                rho = np.dot(X_j, y - y_pred)/n
                self.coef_[j] = self.soft_threshold(rho, self.alpha)
#             print(np.linalg.norm(self.coef_ - beta_old),'--')
            if np.linalg.norm(self.coef_ - beta_old) < self.tol or prev < np.linalg.norm(self.coef_ - beta_old):
                break
            prev = np.linalg.norm(self.coef_ - beta_old)
            
    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
    
    def soft_threshold(self, rho, alpha):
        if rho < -alpha/2:
            return rho + alpha
        elif rho > alpha/2:
            return rho - alpha
        else:
            return 0.0