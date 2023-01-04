import numpy as np


class EarlyStop():
    def __init__(self, patience:float = -1, max_delta:float = 0) -> bool:
        self.patience = patience
        self.max_delta = max_delta
        self.counter = 0
        self.min_val_loss = np.inf
        self.max_val_metric = 0

    def evaluate_loss(self, val_loss):
        if self.patience == -1:
            return False
        
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.max_delta):
            self.counter += 1
            if self.counter > self.patience:
                return True
        
        return False
    
    def evaluate_metric(self, val_metric):
        if self.patience == -1:
            return False
        
        if val_metric > self.max_val_metric:
            self.max_val_metric = val_metric
            self.counter = 0
        elif val_metric < (self.max_val_metric - self.max_delta):
            self.counter += 1
            if self.counter > self.patience:
                return True
        
        return False