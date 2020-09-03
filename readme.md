# ML Model Deployment

A very basic example on how to deploy your Machine Learning models.

The example uses the Iris Dataset. For deployments [Flask](https://flask.palletsprojects.com/en/1.1.x/) and [tornado](https://www.tornadoweb.org/en/stable/) libraries are used.

## Install required libraries
`pip install -r requirements.txt`

## Training the model
`python train_model.py`

## Starting the server
### Flask
`python flask_server.py`
### Tornado
`python tornado_server.py`

## Send Requests to server
`python main.py`

