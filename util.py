from keras.models import load_model
import numpy as np

model_d = load_model("DepressionModel.h5")
model_a = load_model("AnxityModel.h5")
model_s = load_model("StressModel.h5")

def predict_mental_helth(x):
    x = np.array(x)
    x = x.reshape(1,-1)
    d = np.argmax(model_d.predict(x))
    a = np.argmax(model_a.predict(x))
    s = np.argmax(model_s.predict(x))
    
    return d,a,s