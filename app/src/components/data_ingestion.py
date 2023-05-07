import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from app.logger import logging
from app.exception import CustomException
from app.config import Config

from app.src.utils import load_dataframe_from_csv, save_dataframe_to_csv

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join(Config.Z_ARTIFACTS_PATH, 'raw.csv')
    train_data_path: str = os.path.join(Config.Z_ARTIFACTS_PATH, 'train.csv')
    test_data_path: str = os.path.join(Config.Z_ARTIFACTS_PATH, 'test.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Started')
        try:
            data_url = r'https://raw.githubusercontent.com/rahulGarg003/Datasets/main/food-delivery-time-dataset.csv'
            df = load_dataframe_from_csv(file_path=data_url)

            save_dataframe_to_csv(
                dataframe = df,
                file_path=self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info('Train Test Split')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            save_dataframe_to_csv(
                dataframe = train_set,
                file_path=self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            save_dataframe_to_csv(
                dataframe = test_set,
                file_path=self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info('Data Ingestion Completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)