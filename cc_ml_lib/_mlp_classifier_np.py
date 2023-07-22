"""
    implementation of MLP classifier
"""

import numpy as np


class MLPClassifierNP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Initialize weights and biases for hidden layers
        self.hidden_weights = []
        self.hidden_biases = []
        prev_size = input_size
        for size in hidden_sizes:
            self.hidden_weights.append(np.random.randn(size, prev_size))
            self.hidden_biases.append(np.random.randn(size, 1))
            prev_size = size
        
        # Initialize weights and biases for output layer
        self.output_weights = np.random.randn(output_size, prev_size)
        self.output_bias = np.random.randn(output_size, 1)
    
    def forward(self, x):
        pass
    
    def fit(self, X, y, learning_rate=0.01, epochs=100):
        pass
    
    def predict(self, x):
        pass
    
