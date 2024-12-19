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
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
  

class ModelEvaluation:
    def __init__(self):
        self.model_evaluation_config = ModelEvalutionConfig()
        

    
    def initiate_model_evaluation(self,X_train,y_train,X_test,y_test,models):
        try:
            logging.info("model evaluation started")

            report_r2_score = {}
            
            for model_name, model in models.items():

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                MAE, MSE, R2 = evaluate_model(y_test, y_pred)

                
                report_r2_score[model_name] = R2

                logging.info(f"model performance: {model_name} : \n MSE:{MSE} ,\n MAE:{MAE},\n R2:{R2}")
                
            # print(f"model performance: {model_name}")
            # print(f"MSE:{MSE} ,\n MAE:{MAE},\n R2:{R2}" )
            # print("="*20)
            # print("\n")
            best_model_score = max(report_r2_score.values())
            best_model_name = max(report_r2_score)
            logging.info(f"best model: {best_model_name} with score: {best_model_score}")

            best_model_obj = models[best_model_name]
            logging.info(f'save best model object model.pkl file into artifacts')

            save_object(obj=best_model_obj, file_path=self.model_evaluation_config.trained_model_file_path)

            return best_model_name, best_model_score, best_model_obj
            

        except Exception as e:
            error =  CustomException(e, sys)
            raise logging.info(error)
        pass

