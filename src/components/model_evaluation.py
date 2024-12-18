import pandas as pd
import numpy as np
from exception.exception import CustomException 
from logger.logging import logging

import os
import sys
from dataclasses import dataclass
from pathlib import Path # create path class os independent path
from utils.utils import save_object , evaluate_model , load_object



@dataclass
class ModelEvalutionConfig:
    pass
  

class ModelEvalution:
    def __init__(self):
        pass

    
    def initiate_model_evaluation(self):
        try:
            pass
        except Exception as e:
            error =  CustomException(e, sys)
            raise logging.info(error)
        pass

