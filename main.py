import requests
from sklearn.datasets import load_iris
import pandas as pd
import json

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)


def send_request(url):
    d = {"data": df.loc[0:].to_dict(orient="records")}
    print(requests.post(url, data=json.dumps(d)).content)


# send request to Flask server
send_request("http://127.0.0.1:5000/predict")

# send request tot tornado server
send_request("http://127.0.0.1:8000/predict")
