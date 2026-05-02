from fastapi import FastAPI
import torch
import joblib
import numpy as np
from model import ANN
from pydantic import BaseModel
from log import logging

app = FastAPI()

class HouseInput(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: float
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: float
    TAX: float
    PTRATIO: float
    LSTAT: float

# load model
model = ANN()
# model.load_state_dict(torch.load("model.pth"))
feature_names = joblib.load("features.pkl")
# 👇 ye important change
checkpoint = torch.load("model.pth")

model.load_state_dict(checkpoint["model_state"])

model.eval()

# load scaler
scaler = joblib.load("scaler.pkl")



@app.get("/")
def home():
    return {"message": "House Price Prediction API"}

@app.post("/predict")
def predict(data: HouseInput):

    # dict me convert
    input_dict = data.dict()

    logging.info(f"Input: {input_dict}")

    # correct order me convert using feature_names
    x = np.array([[input_dict[feature] for feature in feature_names]])

    # scaling
    x = scaler.transform(x)

    # tensor
    x = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        pred = model(x)

    logging.info(f"Prediction: {pred.item()}")

    return {"prediction": pred.item()}