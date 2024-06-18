from flask import Flask, request, jsonify
import pickle
from utils.utils import preprocess
import sys

sys.path.append("utils")

app = Flask(__name__)

# Load the model
model = None
with open('pickles/final_prediction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json(force=True)

    # preprocess
    data = preprocess(data)

    # Make prediction
    prediction = model.predict(data)[0]
    if prediction == 1:
        return "Survived"
    else:
        return "Passed Away"

if __name__ == '__main__':
    app.run(debug=True, port=8888)