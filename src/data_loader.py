"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for the iris classification dataset."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
    
    def load_iris_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and split iris dataset."""
        logger.info("Loading iris dataset...")
        
        # Load iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Data loaded: train_shape={X_train_scaled.shape}, test_shape={X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_data(self, X_train: np.ndarray, X_test: np.ndarray, 
                  y_train: np.ndarray, y_test: np.ndarray, data_dir: str = "data"):
        """Save processed data to files."""
        import os
        os.makedirs(data_dir, exist_ok=True)
        
        np.save(f"{data_dir}/X_train.npy", X_train)
        np.save(f"{data_dir}/X_test.npy", X_test)
        np.save(f"{data_dir}/y_train.npy", y_train)
        np.save(f"{data_dir}/y_test.npy", y_test)
        
        logger.info(f"Data saved to {data_dir}/")
    
    def load_saved_data(self, data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load previously saved data."""
        X_train = np.load(f"{data_dir}/X_train.npy")
        X_test = np.load(f"{data_dir}/X_test.npy")
        y_train = np.load(f"{data_dir}/y_train.npy")
        y_test = np.load(f"{data_dir}/y_test.npy")
        
        logger.info(f"Data loaded from {data_dir}/")
        return X_train, X_test, y_train, y_test