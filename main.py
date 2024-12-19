from pipeline.training_pipeline import TrainingPipeline
from pipeline.prediction_pipeline import PredictionPipeline ,CustomData


training_pipeline = TrainingPipeline()
best_model_name, best_model_score = training_pipeline.initiate_training_pipeline()
print(best_model_name, best_model_score)


obj = PredictionPipeline()
data = CustomData(carat=1.52, depth=67.0, table=56.0, x=4.0, y=4.0, z=3.0, cut='Premium', color='E', clarity='SI2')
df = data.get_data_as_dataframe()
print(obj.predict(df))