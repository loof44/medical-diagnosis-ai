from flask import Flask, request, jsonify, render_template
import json
import pickle
import numpy as np
from medicalAI import make_prediction


# Import required functions from the original code
# from medicalAI import (
#     label_encoder, symptoms_dict, getDescription, getSeverityDict, getprecautionDict, getInfo, 
#     tree_to_code, DecisionTreeClassifier, columns, train_data, make_prediction
# )

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('base.html')


# Load the trained model
with open('model.pkl', 'rb') as f:
    dt_classifier = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    
    data = json.loads(request.data)
    jsonify({"answer: What symptoms are you experienceing?"})
    message = data["message"]
    jsonify({"answer: how long?"})
    message2 = data["message2"]

    
        

    result = make_prediction(message, message2)
    return jsonify({"answer": result})

if __name__ == "__main__":
    app.run()