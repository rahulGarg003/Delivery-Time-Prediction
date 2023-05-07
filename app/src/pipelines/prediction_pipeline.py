import sys
import os

import numpy as np
import pandas as pd

from app.logger import logging
from app.exception import CustomException
from app.config import Config

from app.src.utils import load_object_from_pkl

class PredictionPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        try:
            preproccesor_obj_path = os.path.join(Config.Z_ARTIFACTS_PATH, 'preprocessor.pkl')
            model_path = os.path.join(Config.Z_ARTIFACTS_PATH, 'model.pkl')
            preproccesor_obj = load_object_from_pkl(preproccesor_obj_path)
            model = load_object_from_pkl(model_path)
            scaled_data = preproccesor_obj.transform(features)
            pred = model.predict(scaled_data)
            return pred[0]
        except Exception as e:
            raise CustomException(e, sys)
        

class CustomData:
    '''
        category_Weather_conditions = ['Sunny', 'Windy', 'Stormy', 'Sandstorms', 'Cloudy', 'Fog']

        category_Road_traffic_density = ['Low', 'Medium', 'High', 'Jam']

        category_Type_of_vehicle = ['scooter', 'electric_scooter', 'motorcycle', 'bicycle']

        category_Festival = ['No', 'Yes']

        category_multiple_deliveries = ['0.0', '1.0', '2.0', '3.0']
    '''
    def __init__(self,
                delivery_person_age: int, 
                delivery_person_ratings: float, 
                vehicle_condition: int,
                weather_conditions: str, 
                road_traffic_density: str, 
                type_of_vehicle: str,
                multiple_deliveries: str, 
                festival: str) -> None:
        self.delivery_person_age = delivery_person_age, 
        self.delivery_person_ratings = delivery_person_ratings, 
        self.vehicle_condition = vehicle_condition,
        self.weather_conditions = weather_conditions, 
        self.road_traffic_density = road_traffic_density, 
        self.type_of_vehicle = type_of_vehicle,
        self.multiple_deliveries = multiple_deliveries, 
        self.festival = festival
    
    def get_data_as_np_array(self):
        try:
            custom_data_input_dict = {
                'Delivery_person_Age' : self.delivery_person_age, 
                'Delivery_person_Ratings' : self.delivery_person_ratings, 
                'Vehicle_condition' : self.vehicle_condition,
                'Weather_conditions' : self.weather_conditions, 
                'Road_traffic_density' : self.road_traffic_density, 
                'Type_of_vehicle' : self.type_of_vehicle,
                'multiple_deliveries' : self.multiple_deliveries, 
                'Festival' : self.festival
            }

            df = pd.DataFrame(custom_data_input_dict)
            return df
        except Exception as e:
            raise CustomException(e, sys)