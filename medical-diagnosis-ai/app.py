from flask import Flask, render_template, request, jsonify
from medicalAI import chat, tree_to_code, trained_classifier, columns
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("base.html")



@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    message = req.get('message')
    num_days = req.get('num_days')

    if message and num_days is not None:
        response = chat(message, num_days)
        return jsonify(response)
    else:
        return jsonify({"answer": "Please provide both 'message' and 'num_days' fields.", "action": "error"})

if __name__ == "__main__":
    app.run(debug=True)
