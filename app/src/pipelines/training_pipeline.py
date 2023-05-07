import os
import sys

from app.logger import logging
from app.exception import CustomException

from app.src.components.data_ingestion import DataIngestion
from app.src.components.data_transformation import DataTransformation
from app.src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self) -> None:
        pass

    def initiate_training_pipeline(self):
        ingestion = DataIngestion()
        train_Data_path, test_data_path = ingestion.initiate_data_ingestion()

        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(
            train_data_path=train_Data_path, 
            test_data_path=test_data_path
        )

        trainer = ModelTrainer()
        trainer.initiate_model_training(
            train_array=train_arr,
            test_array=test_arr
        )