import pickle
import pprint as pp

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel


class TransactionRequest(BaseModel):
    transaction_amount: float
    account_balance: float
    risk_score: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool


app = FastAPI(title="Fraud Detection Prediction Service")

with open("pipeline_v1.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)


def predict_single(transaction) -> float:
    X = dv.transform([transaction])
    result = model.predict_proba(X)[0, 1]
    return float(result)


@app.get("/")
async def docs_redirect():
    return RedirectResponse(url="/docs")


@app.post("/predict")
async def predict(transaction: TransactionRequest) -> PredictionResponse:
    pp.pprint(transaction)
    tx = transaction.model_dump()
    pp.pprint(tx)
    prob = predict_single(tx)
    return PredictionResponse(fraud_probability=prob, is_fraud=(prob >= 0.5))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
