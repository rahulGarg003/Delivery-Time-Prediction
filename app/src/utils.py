import os
import sys
import pickle

import pandas as pd
import numpy as np

from app.logger import logging
from app.exception import CustomException

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def load_dataframe_from_csv(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info(f'data successfully read as pandas dataframe from path: {file_path}') 
        return df
    except Exception as e:
        raise CustomException(e, sys)
    
def save_dataframe_to_csv(dataframe: pd.DataFrame, file_path: str, index: bool, header: bool) -> None:
    try:
        dataframe.to_csv(file_path, index=index, header=header)
        logging.info(f'data successfully saved as csv to path: {file_path}')
    except Exception as e:
        raise CustomException(e, sys)
    
def save_object_as_pkl(file_path: str, obj):
    try:
        with open(file_path, 'wb') as file_obj:
            pickle.dump(
                obj=obj,
                file=file_obj
            )
        logging.info(f'object saved as pkl at: {file_path}')
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object_from_pkl(file_path: str):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models: dict) -> dict:
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            #Training Model
            model.fit(X_train,y_train)

            #predict train data
            y_train_pred = model.predict(X_train)
            #predict test data
            y_test_pred = model.predict(X_test)

            #R2 score for train and test data
            R2score_train = r2_score(y_true=y_train, y_pred=y_train_pred)
            R2score_test = r2_score(y_true=y_test, y_pred=y_test_pred)

            #adjsted R2 score
            adjusted_R2score_train = 1-(((1-R2score_train)*(len(y_train)-1)) / ((len(y_train)-X_train.shape[1]-1)))
            adjusted_R2score_test = 1-(((1-R2score_test)*(len(y_test)-1)) / ((len(y_test)-X_test.shape[1]-1)))

            #Mean Squared Error
            mse_train = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
            mse_test = mean_squared_error(y_true=y_test, y_pred=y_test_pred)

            #Absolute Mean Error
            absolute_mean_error_train = mean_absolute_error(y_true=y_train, y_pred=y_train_pred)
            absolute_mean_error_test = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)

            #Root Mean Squared Error
            rmse_train = np.sqrt(mse_train)
            rmse_test = np.sqrt(mse_test)

            results = {
                'mse_train' : mse_train,
                'mse_test' : mse_test,
                'absolute_mean_error_train' : absolute_mean_error_train,
                'absolute_mean_error_test' : absolute_mean_error_test,
                'rmse_train' : rmse_train,
                'rmse_test' : rmse_test,
                'R2score_train' : R2score_train,
                'R2score_test' : R2score_test,
                'adjusted_R2score_train' : adjusted_R2score_train,
                'adjusted_R2score_test' : adjusted_R2score_test
            }

            report[list(models.keys())[i]] = results
        return report
    except Exception as e:
        raise CustomException(e, sys)
    
if __name__ == '__main__':
    load_dataframe_from_csv('abc')