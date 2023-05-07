import os
import sys
from dataclasses import dataclass

import numpy as np

from app.logger import logging
from app.exception import CustomException
from app.config import Config

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from app.src.utils import load_dataframe_from_csv, save_object_as_pkl

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(Config.Z_ARTIFACTS_PATH, 'preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data transformation Started')

            # defining categorical cols and numerical cols
            categorical_columns = [
                'Weather_conditions', 
                'Road_traffic_density', 
                'Type_of_vehicle',
                'multiple_deliveries', 
                'Festival'
            ]

            numerical_columns = [
                'Delivery_person_Age', 
                'Delivery_person_Ratings', 
                'Vehicle_condition'
            ]

            #defining custom Ranking to each oridinal variable
            category_Weather_conditions = ['Sunny', 'Windy', 'Stormy', 'Sandstorms', 'Cloudy', 'Fog']
            category_Road_traffic_density = ['Low', 'Medium', 'High', 'Jam']
            category_Type_of_vehicle = ['scooter', 'electric_scooter', 'motorcycle', 'bicycle']
            category_Festival = ['No', 'Yes']
            category_multiple_deliveries = ['0.0', '1.0', '2.0', '3.0']

            logging.info('Pipeline Initiated')

            # Numerical Pipeleine
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OrdinalEncoder(
                        categories=[
                            category_Weather_conditions, 
                            category_Road_traffic_density, 
                            category_Type_of_vehicle,
                            category_multiple_deliveries, 
                            category_Festival
                        ]
                    )),
                    ('scaler', StandardScaler())
                ]
            )

            # Preproccessor
            preproccessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            logging.info('Pipeline Completed')
            return preproccessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_data_path: str, test_data_path: str):
        try:
            train_df = load_dataframe_from_csv(file_path=train_data_path)
            test_df = load_dataframe_from_csv(file_path=test_data_path)

            logging.info('Obtaining Preprocessing Object')
            preproccessor_obj = self.get_data_transformation_object()

            target_column = 'Time_taken (min)'
            drop_columns=[
                'ID',
                'Delivery_person_ID', 
                'Restaurant_latitude', 
                'Restaurant_longitude', 
                'Delivery_location_latitude', 
                'Delivery_location_longitude',
                'Order_Date',
                'Time_Orderd',
                'Type_of_order',
                'City',
                'Time_Order_picked',
                target_column
            ]
            train_df_input_features = train_df.drop(
                labels=drop_columns,
                axis=1
            )
            train_df_target_feature = train_df[target_column]

            test_df_input_features = test_df.drop(
                labels=drop_columns,
                axis=1
            )
            test_df_target_feature = test_df[target_column]

            train_df_input_features['multiple_deliveries'] = train_df_input_features['multiple_deliveries'].apply(
                                                                lambda a: str(a)
                                                            ).apply(
                                                                lambda a: np.nan if(a=='nan') else a
                                                            )
            
            test_df_input_features['multiple_deliveries'] = test_df_input_features['multiple_deliveries'].apply(
                                                                lambda a: str(a)
                                                            ).apply(
                                                                lambda a: np.nan if(a=='nan') else a
                                                            )

            logging.info('Applying preproccessing steps on train and test independent features')

            ## transform using preproccessor object
            train_arr_input_features = preproccessor_obj.fit_transform(train_df_input_features)
            test_arr_input_features = preproccessor_obj.transform(test_df_input_features)

            ## converting df to numpy array
            train_arr = np.c_[train_arr_input_features, np.array(train_df_target_feature)]
            test_arr = np.c_[test_arr_input_features, np.array(test_df_target_feature)]

            save_object_as_pkl(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj = preproccessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)