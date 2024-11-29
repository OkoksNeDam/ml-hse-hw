from typing import List
import __main__
import pickle

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from data_transformer_classes import *

__main__.BaseDataTransformer = BaseDataTransformer
__main__.MileageTransformer = MileageTransformer
__main__.EngineTransformer = EngineTransformer
__main__.MaxPowerTransformer = MaxPowerTransformer
__main__.TorqueTransformer = TorqueTransformer
__main__.NameTransformer = NameTransformer

app = FastAPI()

model = pickle.load(open('models/model.pkl', 'rb'))


class Car(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Cars(BaseModel):
    cars: List[Car]


def pydantic_model_to_df(model_instance):
    return pd.DataFrame([jsonable_encoder(model_instance)])


@app.post("/predict_price")
def predict_price(car_info: Car) -> float:
    df_instance = pydantic_model_to_df(car_info)
    prediction = model.predict(df_instance)
    return prediction


@app.post("/predict_prices")
def predict_prices(cars_info: Cars) -> List[float]:
    df_instances = list(map(pydantic_model_to_df, cars_info.cars))
    df_instances = pd.concat(df_instances)
    predictions = model.predict(df_instances)
    return predictions
