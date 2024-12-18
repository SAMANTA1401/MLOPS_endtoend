import pandas as pd
import numpy as np
from src.exception.exception import CustomException 
from src.logger.logging import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path # create path class os independent path
from src.utils.utils import save_object , evaluate_model
from sklearn.linear_model import LinearRegression , Lasso , Ridge 



@dataclass
class ModelTrainerConfig:
    pass
  

class ModelTrainer:
    def __init__(self):
        pass

    
    def initiate_model_training(self):
        try:
            pass
        except Exception as e:
            error =  CustomException(e, sys)
            raise logging.info(error)
        pass

