import pandas as pd
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("prophet_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        date = request.form["date"]
        if date:
            future = pd.DataFrame({"ds": [pd.to_datetime(date)]})

            
            forecast = model.predict(future)
            daily_prediction = forecast["yhat"].iloc[0]  

            hourly_prediction = round(daily_prediction / 24, 2)  

            prediction = hourly_prediction 

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
