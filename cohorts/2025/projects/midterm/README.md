# Machine Learning Zoomcamp (Cohort 2025)

## Midterm Project: Fraud Detection Prediction Service

### Description of the problem

This project is based on a public domain dataset published by [Samay Ashar](https://www.kaggle.com/samayashar) on [Kaggle.com](https://www.kaggle.com/datasets/samayashar/fraud-detection-transactions-dataset). It was named the _**Fraud Detection Transactions Dataset**_ and it is described as a _high-quality synthetic dataset for fraud detection using XGBoost_.

This dataset is designed to help machine learning enthusiasts develop robust fraud detection models. It contains realistic synthetic transaction data, including user information, transaction types, risk scores, and more, making it ideal for binary classification tasks with classification algorithms (including non-XGBoost models).

#### Key Characteristics of the Dataset:

- It contains **21 features** that capture various aspects of a financial transaction.
- It has a realistic structure, including numerical, categorical, and temporal data.
- The target variable, `fraud_label`, is **binary** (0 = Not Fraud, 1 = Fraud).
- It is optimized for high accuracy with models like XGBoost and other machine learning algorithms.
- The dataset is useful for tasks such as anomaly detection, risk analysis, and security research.

The primary goal of this project is to apply fundamental machine learning methodologies to create a **predictive fraud detection service**, serving educational purposes.

### Repository contents

This repository is a fork of the **`#mlzoomcamp`** course. The project was implemented at this folder: [cohorts/2025/projects/midterm](.). This folder is organized to support the development, training, and deployment of the fraud detection model.

Below is an overview of the key directories and files:

- [README.md](README.md): This file, providing an overview of the project.
- [pyproject.toml](pyproject.toml): The `uv` managed project file.
- [uv.lock](uv.lock): The real versions of the dependencies used here.
- [train.py](train.py): The training script.
- [predict.py](predict.py): The predictive **FastAPI** application.
- [pipeline_v1.bin](pipeline_v1.bin): The serialized trained model.
- [Dockerfile](Dockerfile): The deployment docker file.
- [run-docker.sh](run-docker.sh): A script to run "docker build" & "docker run" locally.
- [submission.ipynb](submission.ipynb): The rest of the data cleaning, data analysis and predictive modelling stuff.
- [fraud-detection-transactions-dataset.zip](fraud-detection-transactions-dataset.zip): A saved copy of the data set file downloaded from Kaggle.

### How to run this project

This project was created using **Github's Codespaces**. To run any artifact create a free codespace from the repository's main page. Then from your browser: open a terminal view and change current directory to:

```
$ cd cohorts/2025/projects/midterm
```

Now, from this directory, you can run:

- `$ uv run train.py` # to train the model with local data file
- `$ uv run predict.py` # to test the prediction microservice with local `pipeline_v1.bin`
- `$ ./run-docker.sh` # to create a docker image of the service and run it locally

To test the service, deployed at **Google Cloud Run** visit the documentation page at:

- <https://machine-learning-zoomcamp-428800185377.europe-west1.run.app>

Or run from any Python session (e.g. `uv run python`) the following code:

```
import requests

transaction = {
    "transaction_amount": 200,
    "account_balance": 0,
    "daily_transaction_count": 0,
    "avg_transaction_amount_7d": 15,
    "failed_transaction_count_7d": 0,
    "card_age": 0,
    "transaction_distance": 1000,
    "risk_score": 0.2,
    "authentication_method": "Password",
    "card_type": "Visa",
    "device_type": "Mobile",
    "hour_of_day": 10,
    "ip_address_flag": 0,
    "is_weekend": 0,
    "location": "New York",
    "merchant_category": "Electronics",
    "previous_fraudulent_activity": 0,
    "transaction_type": "Online",
}

res = requests.post("https://machine-learning-zoomcamp-428800185377.europe-west1.run.app/predict", json=transaction)
res.raise_for_status()

print(res.json())
```

More information about deployment can be found at [submission.ipynb](submission.ipynb).
