import pickle
import pprint as pp
from pydantic import BaseModel


from fastapi import FastAPI
import uvicorn


class Customer(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool


app = FastAPI(title="customer-churn-prediction")

with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer) -> PredictResponse:
    pp.pprint(customer)
    c = customer.model_dump()
    pp.pprint(c)
    prob = predict_single(c)

    return PredictResponse(churn_probability=prob, churn=prob >= 0.5)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
