import pandas as pd
import numpy as np
from src.exception.exception import CustomException 
from src.logger.logging import logging

import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path # create path class os independent path



@dataclass
class DataIngestionConfig:
    pass
  

class DataIngestion:
    def __init__(self):
        pass

    
    def initiate_data_ingestion(self):
        try:
            pass
        except Exception as e:
            error =  CustomException(e, sys)
            raise logging.info(error)
        pass

