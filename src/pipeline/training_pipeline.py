import os
import sys
from exception.exception import CustomException
from logger.logging import logging
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer
from components.model_evaluation import ModelEvaluation


logging.info("Starting training pipeline ")


data_ingestion_obj = DataIngestion()
train_data_path, test_data_path = data_ingestion_obj.initiate_data_ingestion()

logging.info("Data ingestion pipeline completed ")

data_transformation_obj = DataTransformation()
train_arr, test_arr, _ = data_transformation_obj.initiate_data_transformation(train_data_path, test_data_path)

logging.info("Data transformation pipeline completed ")

model_trainer_obj = ModelTrainer()
X_train, y_train, X_test, y_test, models = model_trainer_obj.initiate_model_training(train_arr, test_arr)

logging.info("Model training pipeline completed ")

model_evaluation_obj = ModelEvaluation()
best_model_name, best_model_score, best_model_obj = model_evaluation_obj.initiate_model_evaluation(X_train, y_train, X_test, y_test, models)

logging.info("Model evaluation pipeline completed ")

logging.info(f"best model object {best_model_name} as model.pkl  ready to use")


# python src/pipeline/training_pipeline.py