import pandas as pd
import numpy as np
from src.exception.exception import CustomException 
from src.logger.logging import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path # create path class os independent path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler , OrdinalEncoder 

from src.utils.utils import save_object



@dataclass
class DataTransformationConfig:
    pass
  

class DataTransformation:
    def __init__(self):
        pass

    
    def initiate_data_transformation(self):
        try:
            pass
        except Exception as e:
            error =  CustomException(e, sys)
            raise logging.info(error)
        pass

