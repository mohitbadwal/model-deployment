import tornado.ioloop
import tornado.web
import json
import urllib
import pandas as pd
import joblib as jb

# fix for running tornado in python 3.8 and windows 10
import sys, asyncio

if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# make a class which inherits tornado.web.RequestHandler class
class MainHandler(tornado.web.RequestHandler):

    def get(self):
        self.write("Hello World!")

    def post(self):
        # check if request has a body
        if self.request.body:

            # read json data which was sent in the request body
            try:
                json_data = json.loads(self.request.body)
            except:
                json_data = json.loads(urllib.parse.unquote_plus(self.request.body))

            # convert json data to pandas dataframe
            df = pd.DataFrame(json_data["data"])

            # predict
            predictions = model.predict(df)
            print(predictions)

            # send back the predictions
            self.write(json.dumps({"predictions": predictions.tolist()}))


def make_app():
    return tornado.web.Application([("/predict", MainHandler)], debug=True)


if __name__ == "__main__":
    model = jb.load("models/iris_model.pkl")
    print("Model Reading Successful")
    port = 8000
    app = make_app()
    app.listen(port)
    print("Server starting in http://127.0.0.1:"+str(port))
    tornado.ioloop.IOLoop.current().start()
