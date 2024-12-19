import os
import sys
from exception.exception import CustomException
from logger.logging import logging
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer
from components.model_evaluation import ModelEvaluation



class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_obj = DataIngestion()
        self.data_transformation_obj = DataTransformation()
        self.model_trainer_obj = ModelTrainer()
        self.model_evaluation_obj = ModelEvaluation()

    def initiate_training_pipeline(self):
        try:
            logging.info("Starting training pipeline ")
            train_data_path, test_data_path = self.data_ingestion_obj.initiate_data_ingestion()
            logging.info("Data ingestion pipeline completed ")
            train_arr, test_arr, _ = self.data_transformation_obj.initiate_data_transformation(train_data_path, test_data_path)
            logging.info("Data transformation pipeline completed ")
            X_train, y_train, X_test, y_test, models = self.model_trainer_obj.initiate_model_training(train_arr, test_arr)
            logging.info("Model training pipeline completed ")
            best_model_name, best_model_score, _ = self.model_evaluation_obj.initiate_model_evaluation(X_train, y_train, X_test, y_test, models)
            logging.info("Model evaluation pipeline completed ")
            logging.info(f"best model object {best_model_name} as model.pkl  ready to use")

            return best_model_name, best_model_score

        except CustomException as e:
            error = CustomException(e, sys)
            raise logging.info(error)



if __name__ == '__main__':
    training_pipeline = TrainingPipeline()
    best_model_name, best_model_score = training_pipeline.initiate_training_pipeline()
    print(best_model_name, best_model_score)



# python src/pipeline/training_pipeline.py