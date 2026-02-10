import xgboost as xgb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from tabpfn import TabPFNClassifier

class MCDropoutNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3), # Increased dropout for better uncertainty quantification
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

from sklearn.preprocessing import StandardScaler

class UncertaintyModel(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=200, lr=0.001):
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        # Convert pandas/numpy to tensor
        X_arr = X.values if hasattr(X, 'values') else X
        y_arr = y.values if hasattr(y, 'values') else y
        
        # Scale Data (Crucial for MLP)
        X_scaled = self.scaler.fit_transform(X_arr)
        
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_arr).reshape(-1, 1)
        
        self.model = MCDropoutNetwork(X_arr.shape[1])
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            out = self.model(X_tensor)
            loss = criterion(out, y_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict_uncertainty(self, X, n_samples=100):
        """
        Returns (mean_probability, std_deviation)
        """
        if self.model is None:
            raise ValueError("Model not fitted")
            
        self.model.train() # CRITICAL: Enable dropout during inference
        
        X_arr = X.values if hasattr(X, 'values') else X
        # Apply Scaling
        X_scaled = self.scaler.transform(X_arr)
        X_tensor = torch.FloatTensor(X_scaled)
        
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.model(X_tensor).numpy())
                
        preds = np.array(preds) # Shape: (n_samples, n_instances, 1)
        
        # Mean prediction (probability of class 1)
        mean_pred = preds.mean(axis=0).flatten()
        # Uncertainty (standard deviation of probability)
        std_pred = preds.std(axis=0).flatten()
        
        return mean_pred, std_pred

def get_tabpfn():
    # TabPFN
    return TabPFNClassifier(device='cpu')

def get_xgboost():
    return xgb.XGBClassifier(eval_metric='logloss', random_state=42)
