import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.exception.exception import CustomException
from src.logger.logging import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        error =  CustomException(e, sys)
        raise logging.info(error)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        error =  CustomException(e, sys)
        raise logging.info(error)
    

def evaluate_model(true, predicted):
    try:
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        # rmse = np.sqrt(mean_squared_error(true, predicted))
        r2_square = r2_score(true, predicted)
        return mae, mse, r2_square
    except Exception as e:
        error =  CustomException(e, sys)
        raise logging.info(error)