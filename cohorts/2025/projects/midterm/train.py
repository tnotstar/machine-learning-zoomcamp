from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score

import warnings
import pickle

import numpy as np
import pandas as pd


df = pd.read_csv("fraud-detection-transactions-dataset.zip")
df.columns = df.columns.str.lower()

for feature in df.select_dtypes(include="object").columns:
    df[feature] = df[feature].astype("category")

for feature in [
    "ip_address_flag",
    "previous_fraudulent_activity",
    "is_weekend",
    "fraud_label",
]:
    df[feature] = df[feature].astype("category")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour_of_day"] = df["timestamp"].dt.hour
df["hour_of_day"] = df["hour_of_day"].astype("category")


for feature in ["transaction_id", "user_id", "timestamp"]:
    del df[feature]

target_variable = "fraud_label"
print(f"Target variable: '{target_variable}'")

numerical_features = df.select_dtypes(include=np.number).columns
if target_variable in numerical_features:
    numerical_features = numerical_features.drop(target_variable)
print(f"Numerical features: {sorted(numerical_features.tolist())}")

categorical_features = df.select_dtypes(include="category").columns
if target_variable in categorical_features:
    categorical_features = categorical_features.drop(target_variable)
print(f"Categorical features: {sorted(categorical_features.tolist())}")

all_features = categorical_features.append(numerical_features)

common_y_name = target_variable
common_random_state = 11562788

df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=common_random_state
)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=common_random_state
)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(f"Length of the train dataset: {len(df_train)}")
print(f"Length of the validation dataset: {len(df_val)}")
print(f"Length of the test dataset: {len(df_test)}")


def split_y_X(df, y_name):
    y = df[y_name]
    X = df.drop(columns=[y_name])
    return y, X


y_full_train, X_full_train = split_y_X(df_full_train, common_y_name)
y_train, Xtmp_train = split_y_X(df_train, common_y_name)
y_val, Xtmp_val = split_y_X(df_val, common_y_name)
y_test, Xtmp_test = split_y_X(df_test, common_y_name)


dv = DictVectorizer(sparse=False)

train_dict = Xtmp_train.to_dict(orient="records")
X_train = dv.fit_transform(train_dict)
print(f"Training features: {X_train.shape}")

model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
model.fit(X_train, y_train)
print(f"Model: {model}")

val_dict = Xtmp_val.to_dict(orient="records")
X_val = dv.transform(val_dict)
print(f"Validation features: {X_val.shape}")

y_pred = model.predict_proba(X_val)[:, 1]

roc_auc = round(roc_auc_score(y_val, y_pred >= 0.5), 3)
print(f"AUC: {roc_auc}")

output_file = "pipeline_v1.bin"
with open(output_file, "wb") as output:
    pickle.dump((dv, model), output)  # type: ignore
print(f"Model saved to {output_file}")


input_file = "pipeline_v1.bin"
with open(input_file, "rb") as input:
    dv, model = pickle.load(input)
print(f"Model read from {input_file}")


def predict_single(transaction) -> float:
    X = dv.transform([transaction])
    result = model.predict_proba(X)[0, 1]
    return float(result)


transaction = {
    "transaction_amount": 200,
    "account_balance": 0,
    "daily_transaction_count": 0,
    "avg_transaction_amount_7d": 15,
    "failed_transaction_count_7d": 8,
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

y_pred = predict_single(transaction)

print("input:", transaction)
print("output:", y_pred)
