import pandas as pd
import numpy as np
from exception.exception import CustomException 
from logger.logging import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path # create path class os independent path
from utils.utils import save_object , evaluate_model
from sklearn.linear_model import LinearRegression , Lasso , Ridge ,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
  

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
       
    def initiate_model_training(self, train_array, test_array):
        logging.info("Model training initiated")
        try:
            logging.info("splitting train and test data into X and y")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet(),
                'RandomForestRegressor': RandomForestRegressor(),
                'XGBRegressor': XGBRegressor()
            }

            return X_train, y_train, X_test, y_test, models

            
        except Exception as e:
            error =  CustomException(e, sys)
            raise logging.info(error)
        

