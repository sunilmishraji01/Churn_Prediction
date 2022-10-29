import pickle
from flask import Flask, request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app =Flask(__name__)
# load the model
with open("saved__model/tree_clf_V0", "rb") as f:
    tree_clf_V0=pickle.load(f)


@app.route("/")
def home():
    return render_template("home.html")


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    output=tree_clf_V0.predict(final_input)[0]
    return render_template("home.html",prediction_text=" Churn Prediction is {}".format(output))


if __name__=="__main__":
    app.run(debug=True)