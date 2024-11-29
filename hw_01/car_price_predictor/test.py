import json

import pandas as pd
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_predict_price():
    data_index = 0
    car_info = pd.read_csv("test_data/test_data.csv").iloc[data_index].to_json()
    response = client.post("/predict_price", content=car_info)
    assert response.status_code == 200
    assert response.json() == 404661.34019994736


def test_predict_prices():
    data_indices = [0, 1, 2]

    cars_info = pd.read_csv("test_data/test_data.csv").iloc[data_indices].to_json(orient='records')
    cars_info = json.loads(cars_info)
    response = client.post("/predict_prices", content=json.dumps({"cars": cars_info}))
    assert response.status_code == 200
    assert response.json() == [404661.34019994736, 705877.3692527562, 348741.23347622156]