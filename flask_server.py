from flask import Flask
from flask import request
import joblib as jb
import json
import pandas as pd

app = Flask(__name__)


# this function will trigger when someone sends a request to /predict
@app.route('/predict', methods=["GET", "POST"])
def hello():
    # if request method is GET
    if request.method == "GET":
        return "Hello World!"

    # if request method is POST
    elif request.method == "POST":

        # read json data which was sent in the request body
        json_data = request.get_json(force=True)
        print(json_data)

        # convert json data to pandas dataframe
        df = pd.DataFrame(json_data['data'])

        # predict
        predictions = model.predict(df)
        print(predictions)

        # send back the predictions
        return json.dumps({"predictions": predictions.tolist()})


if __name__ == '__main__':
    model = jb.load("models/iris_model.pkl")
    print("Model Reading Successful")
    app.run()
