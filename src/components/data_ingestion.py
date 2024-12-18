import pandas as pd
import numpy as np
from exception.exception import CustomException 
from logger.logging import logging

# src is package and directories inside package are module we need to import module from the package not package

import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path # create path class os independent path

# artifacts inside the current working directory that is os.getcwd()

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join('artifacts','raw.csv')
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
  

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
   

    
    def initiate_data_ingestion(self):
        logging.info('data ingestion initiated')
        try:
            csv_path = os.path.join(os.getcwd(),'artifacts','cubic_zirconia.csv')
            data = pd.read_csv(csv_path)
            logging.info('data loaded successfully')
            os.makedirs(os.path.dirname(os.path.join(self.config.raw_data_path)), exist_ok=True)
            data.to_csv(self.config.raw_data_path, index=False)
            logging.info('data saved to raw data path')
            train_data, test_data = train_test_split(data, test_size=0.25)
            logging.info('data splitted successfully')
            os.makedirs(os.path.dirname(os.path.join(self.config.train_data_path)), exist_ok=True)
            train_data.to_csv(self.config.train_data_path, index=False)
            logging.info('train data saved to train data path')
            os.makedirs(os.path.dirname(os.path.join(self.config.test_data_path)), exist_ok=True)
            test_data.to_csv(self.config.test_data_path, index=False)
            logging.info('test data saved to test data path')
            logging.info('data ingestion completed successfully')

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )
            


        except Exception as e:
            error =  CustomException(e, sys)
            raise logging.info(error)
        

# python  src/components/data_ingestion.py
# python src.components.data_ingestion.py
if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print(f'train data path: {train_data_path}')
    print(f'test data path: {test_data_path}')
    # print(os.getcwd())
    # csv_path = os.path.join(os.getcwd(),'artifacts','cubic_zirconia.csv')
    # print(csv_path)

    