#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import load_model


# In[3]:


model_d = load_model("DepressionModel.h5")
model_a = load_model("AnxityModel.h5")
model_s = load_model("StressModel.h5")


# In[4]:


pickle.dump(model_d,open("DepressionModel.pkl","wb"))
pickle.dump(model_a,open("AnxityModel.pkl","wb"))
pickle.dump(model_s,open("StressModel.pkl","wb"))


# In[1]:


def predict_mental_helth(x):
    x = np.array(x)
    x = x.reshape(1,-1)
    d = np.argmax(model_d.predict(x))
    a = np.argmax(model_a.predict(x))
    s = np.argmax(model_s.predict(x))
    
    return d,a,s


# In[5]:


from flask import Flask,request
#from util import predict_mental_helth

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



app.run(debug=True,host="0.0.0.0")


# In[ ]:





# In[ ]:




