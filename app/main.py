from datetime import datetime
import logging
import os
from typing import List, Optional

from fastapi import FastAPI
import numpy as np
import pandas as pd
from pydantic import BaseModel

import uvicorn

logger = logging.getLogger("model_deaths")

v1_path = "/v1"
model_path = "/deaths/confirmed"

logger.info("Loading API")
app = FastAPI(
    title="COVID-19 Deaths Prediction",
    description="This API exposes predictions of the number of deaths.",
    version="0.1.1",
    openapi_url=f"{v1_path}{model_path}/openapi.json",
    docs_url=f"{v1_path}{model_path}/docs",
    redoc_url=f"{v1_path}{model_path}/redoc",
)

predictions_path = os.getenv("PREDICTIONS_PATH", "app/predictions.pkl")
logger.info(f"Loading predictions from {predictions_path}")
predictions = pd.read_pickle(predictions_path)


class Prediction(BaseModel):
    timestamp: datetime
    deaths_prediction: np.float64
    cumulative_deaths_prediction: np.float64
    residuals_low: np.float64
    residuals_high: np.float64
    days_since_100_cases: int
    deaths: Optional[float]
    cumulative_deaths: Optional[float]

    def __init__(self, **data) -> None:
        """Custom init to parse datetime64"""
        if isinstance(data["timestamp"], pd.Timestamp):
            data["timestamp"] = data["timestamp"].to_pydatetime()
        super().__init__(**data)


class PredictionsOut(BaseModel):
    country: str
    result: List[Prediction]


@app.get(v1_path + model_path + "/countries/{country}")
def read_root(
    country: str,
    time_from: datetime = datetime(2019, 1, 1),
    time_to: datetime = datetime(2222, 1, 1),
) -> PredictionsOut:
    df = predictions[predictions["country"] == country]
    df = df[(df.index >= time_from) & (df.index < time_to)]
    df = df.where(pd.notnull(df), None)  # Make nans a None so that it encodes properly
    return PredictionsOut(
        country=country,
        result=list(map(lambda x: Prediction(**x), df.to_dict(orient="records"))),
    )


@app.get(f"{v1_path}{model_path}/countries")
def read_item():
    return predictions["country"].unique().tolist()


@app.get(f"{v1_path}{model_path}/health")
def redirect():
    return {"detail": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
