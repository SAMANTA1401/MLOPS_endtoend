import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from exception.exception import CustomException
from logger.logging import logging
from utils.utils import load_object

@dataclass
class  PredictionPipelineConfig():
    preprocessor_path:str = os.path.join('artifacts', 'preprocessor.pkl')
    model_path:str = os.path.join('artifacts', 'model.pkl')


class PredictionPipeline:
    def __init__(self):
        self.prediction_pipeline_config = PredictionPipelineConfig()

    def predict(self, features): # freature must be in dataframe or 2d array [[]] for single prediction [[],[]] for list prediction 
        logging.info("Prediction pipeline initiated")
        try:
            preprocessor = load_object(self.prediction_pipeline_config.preprocessor_path)
            model = load_object(self.prediction_pipeline_config.model_path)

            data_scaled = preprocessor.transform(features)
            logging.info(f'preprocessed data: \n{data_scaled}')
            pred = model.predict(data_scaled)
            logging.info(f'Predicted price is : {pred}')

            return pred

        except Exception as e:
            error = CustomException(e, sys)
            raise logging.info(error)
        

class CustomData:
    def __init__(self, carat:float, depth:float, table:float, x:float, y:float, z:float, cut:str, color:str, clarity:str):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info(f'Dataframe Created: \n{df.to_string()}')

            return df
        
        except Exception as e:
            error = CustomException(e, sys)
            raise logging.info(error)

        

if __name__ == '__main__':
    obj = PredictionPipeline()
    data = CustomData(1.52, 67.0, 56.0, 4.0, 4.0, 3.0, 'Premium', 'E', 'SI2')
    df = data.get_data_as_dataframe()
    print(obj.predict(df))