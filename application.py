from app.src.pipelines.training_pipeline import TrainingPipeline

from app.src.pipelines.prediction_pipeline import CustomData, PredictionPipeline


if __name__ == '__main__':
    # pipeline = TrainingPipeline()
    # pipeline.initiate_training_pipeline()
    data = CustomData(
        delivery_person_age=32,
        delivery_person_ratings=4.5,
        vehicle_condition=0,
        weather_conditions='Windy',
        road_traffic_density='Low',
        type_of_vehicle='scooter',
        multiple_deliveries='1.0',
        festival='Yes'
    )

    prediction = PredictionPipeline()
    print(prediction.predict(data.get_data_as_np_array()))