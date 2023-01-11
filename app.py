from flask import Flask,request
from util import predict_mental_helth

import json

app = Flask(__name__)

index_to_class= {
            0:"Extremly Sevear",
            1:"Mild",
            2:"Modarate",
            3:"Normal",
            4:"Sevear"
        }

@app.route("/predict",methods=["GET"])
def predict():
    x = []
    for i in request.args.getlist("one"):
        x.append(int(i))
    d,a,s = predict_mental_helth(x)
    return json.dumps({"Depression":index_to_class[d],
              "Anxity":index_to_class[a],
              "Stress":index_to_class[s]})

# @app.route("/",methods=["GET"])
# def home():
#     return json.dumps({"Hello":"World it is working now wow!!!"})


if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)