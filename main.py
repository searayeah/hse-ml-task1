import io
from typing import List

import __main__
import dill as pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from loguru import logger

__main__.pd = pd
__main__.np = np

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
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


class ItemFile(BaseModel):
    file: UploadFile


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame.from_dict([item.model_dump()])
    X = data.drop("selling_price", axis=1)
    y = data["selling_price"]
    pred = model.predict(X)
    return pred


@app.post("/predict_items")
def predict_items(file: UploadFile) -> FileResponse:
    contents = file.file.read()
    data = pd.read_csv(io.BytesIO(contents))
    file.file.close()
    X = data.drop("selling_price", axis=1)
    y = data["selling_price"]
    pred = model.predict(X)
    logger.info(f"R2: {r2_score(y, pred)}")
    logger.info(f"MSE: {MSE(y, pred)}")
    data["predict"] = pred
    data.to_csv("predict.csv", index=False)
    response = FileResponse(
        path="predict.csv", media_type="text/csv", filename="predicted.csv"
    )
    return response
