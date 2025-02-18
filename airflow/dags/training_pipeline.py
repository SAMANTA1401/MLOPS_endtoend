from __future__ import annotations
import json
from textwrap import dedent
from datetime import datetime
# import pendulum
# from airflow import settings
# from pendulum import tz
# Access the timezone function from the pendulum.tz module
# settings.TIMEZONE = tz.timezone("UTC")
from airflow import DAG
from airflow.operators.python import PythonOperator
from pipeline.training_pipeline import TrainingPipeline
from airflow.decorators import task

training_pipeline=TrainingPipeline()

with DAG(
    "gemstone_training_pipeline",
    default_args={"retries": 2},
    description="it is my training pipeline",
    schedule="@weekly",# here you can test based on hour or mints but make sure here you container is up and running
    start_date=datetime(2024, 12, 20),
    # schedule="0 0 * * 0",
    catchup=False,
    tags=["machine_learning ","classification","gemstone"],
) as dag:
    
    dag.doc_md = __doc__
    

    def data_ingestion(**kwargs):
        ti = kwargs["ti"]
        train_data_path,test_data_path=training_pipeline.start_data_ingestion()
        ti.xcom_push("data_ingestion_artifact", {"train_data_path":train_data_path,"test_data_path":test_data_path})

    def data_transformations(**kwargs):
        ti = kwargs["ti"]
        data_ingestion_artifact=ti.xcom_pull(task_ids="data_ingestion",key="data_ingestion_artifact") 
        train_arr,test_arr=training_pipeline.start_data_transformation(data_ingestion_artifact["train_data_path"],data_ingestion_artifact["test_data_path"])
        train_arr=train_arr.tolist()
        test_arr=test_arr.tolist()
        ti.xcom_push("data_transformations_artifcat", {"train_arr":train_arr,"test_arr":test_arr})

    def model_trainer(**kwargs):
        import numpy as np
        ti = kwargs["ti"]
        data_transformation_artifact = ti.xcom_pull(task_ids="data_transformation", key="data_transformations_artifcat")
        train_arr=np.array(data_transformation_artifact["train_arr"])
        test_arr=np.array(data_transformation_artifact["test_arr"])
        training_pipeline.start_model_training(train_arr,test_arr)

    def model_evaluation(**kwargs):
        ti = kwargs["ti"]
        data_transformation_artifact = ti.xcom_pull(task_ids="data_transformation", key="data_transformations_artifcat")
        train_arr=data_transformation_artifact["train_arr"]
        test_arr=data_transformation_artifact["test_arr"]
        training_pipeline.start_model_evaluation(train_arr, test_arr)
    
    ## you have to config azure blob or s3
    def push_data_to_s3(**kwargs):
        import os
        bucket_name=os.getenv("BUCKET_NAME")
        artifact_folder="/app/artifacts"
        os.system(f"aws s3 sync {artifact_folder} s3:/{bucket_name}/artifact")
        
        
        
    data_ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=data_ingestion,
    )
    data_ingestion_task.doc_md = dedent(
        """\
    #### Ingestion task
    this task creates a train and test file.
    """
    )

    data_transform_task = PythonOperator(
        task_id="data_transformation",
        python_callable=data_transformations,
    )
    data_transform_task.doc_md = dedent(
        """\
    #### Transformation task
    this task performs the transformation
    """
    )

    model_trainer_task = PythonOperator(
        task_id="model_trainer",
        python_callable=model_trainer,
    )
    model_trainer_task.doc_md = dedent(
        """\
    #### model trainer task
    this task perform training
    """
    )

    model_evaluation_task = PythonOperator(
        task_id="model_evaluation",
        python_callable=model_evaluation,
    )
    model_evaluation_task.doc_md = dedent(
        """\
        ### model evaluation task
        this task performing evaluation
        """
    )

    
   
    # push_data_to_s3_task = PythonOperator(
    #     task_id="push_data_to_s3",
    #     python_callable=push_data_to_s3
    #     )


data_ingestion_task >> data_transform_task >> model_trainer_task # >> push_data_to_s3_task