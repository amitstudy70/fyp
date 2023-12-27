from flask import Flask, request, render_template
import numpy as np
import pandas
import sklearn
import pickle
import math

# importing model
model = pickle.load(open("model.pkl", "rb"))
# sc = pickle.load(open("standscaler.pkl", "rb"))
# ms = pickle.load(open("minmaxscaler.pkl", "rb"))

# creating flask app
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    N = float(request.form["Nitrogen"])
    P = float(request.form["Phosporus"])
    K = float(request.form["Potassium"])
    temp = float(request.form["Temperature"])
    humidity = float(request.form["Humidity"])
    ph = float(request.form["Ph"])
    # rainfall = float(request.form["Rainfall"])

    # Format numerical values to 8 decimal places
    N = round(N, 12)
    P = round(P, 12)
    K = round(K, 12)
    temp = round(temp, 12)
    humidity = round(humidity, 12)
    ph = round(ph, 12)
    # rainfall = round(rainfall, 12)

    feature_list = [N, P, K, temp, humidity, ph]
    single_pred = np.array(feature_list).reshape(1, -1)

    # scaled_features = ms.transform(single_pred)
    # final_features = sc.transform(scaled_features)
    prediction = model.predict(single_pred)

    crop_dict = {
        1: "Mango",
        2: "Rose",
        3: "Aloevera",
        4: "Curry Leaves",
        5: "Lemon",
        6: "Jasmine",
        7: "Organic Taro Leaves",
        8: "Mud Apple",
        9: "Coriander",
        10: "Tomato",
        11: "Spinach",
        12: "Fenugreek",
        13: "Green Chilli",
        14: "Basil",
        15: "Radish",
        16: "Carrot"
    }

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template("index.html", result=result)


# python main
if __name__ == "__main__":
    app.run(debug=True)
