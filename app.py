import sys
import json
import numpy as np
import pickle as pkl
from flask import Flask, request, jsonify, render_template
app = Flask(_name_)
# Load the model
dtr = pkl.load(open('regmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    input_data = np.array(list(data.values())).reshape(1, -1)
    output = dtr.predict(input_data)
    return jsonify(output.tolist())  # Convert numpy array to a Python list

@app.route('/predict',methods =['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1, -1)
    print(final_input)
    output = dtr.predict(final_input)[0]
    return render_template("home.html",prediction_text = "The internal feasibility value is {}".format(output))


if _name_ == "_main_":
    app.run(debug=True)