import pandas as pd
import numpy as np
from exception.exception import CustomException 
from logger.logging import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path # create path class os independent path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler , OrdinalEncoder 

from utils.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

  

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        try:
            logging.info('preprocessing object initiated')

            categorical_columns = ['cut', 'color', 'clarity']
            numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']

            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('pipeline initiated')
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])
            logging.info('preprocessor object created')
            return preprocessor

        except Exception as e:
            error = CustomException(e, sys)
            raise logging.info(error)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info('data transformation initiated')
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info('read train and test data')
            logging.info(f'Train Dataframe Head : \n{train_data.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_data.head().to_string()}')

            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']
            input_feature_train_df = train_data.drop(drop_columns, axis=1)
            target_train_df = train_data[target_column_name]

            input_feature_test_df = test_data.drop(drop_columns, axis=1)
            target_test_df = test_data[target_column_name]
            logging.info("drop id, separate target column from train and test data")

            preprocessor_obj = self.get_data_transformer_object()
            logging.info('applying preprocessing object on training and testing datasets')

            preprocessor_obj.fit_transform(input_feature_train_df)
            preprocessor_obj.transform(input_feature_test_df)
            logging.info('train and test data preprocessed successfully')

            train_arr = np.c_[input_feature_train_df, np.array(target_train_df)] # np.c_ concat axis=1
            test_arr = np.c_[input_feature_test_df, np.array(target_test_df)]
            logging.info('train and test data transformed to numpy arrays and concatenated')

            
            save_object(obj=preprocessor_obj, file_path=self.data_transformation_config.preprocessor_obj_file_path)
            logging.info('save preprocessor object to artifacts folder')

            logging.info('data transformation completed successfully')
            return train_arr, test_arr, preprocessor_obj


        except Exception as e:
            error =  CustomException(e, sys)
            raise logging.info(error)
        

if __name__ == '__main__':
    data_transformation = DataTransformation()
    train_path =os.path.join('artifacts','train.csv')
    test_path = os.path.join('artifacts','test.csv')
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)
    print(train_arr[:5])
    print(test_arr[:5])

