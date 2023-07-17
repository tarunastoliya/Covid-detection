from flask import Flask, render_template, request
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras


app = Flask(__name__)

model = keras.models.load_model("D:\My Projects\covid-detection-cnn-master\covid-detection-cnn-master\inceptionv3_chest.h5")
print("+"*50, "Model is loaded")

# labels = pd.read_csv("labels.txt").values


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def prediction():

    img = request.files['img']

    img.save("img.jpg")

    image = cv2.imread("img.jpg")

    # arrange format as per keras)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255

    # image = np.reshape(image, (1,224,224,3))
    image = np.expand_dims(image, axis=0)

    pred = model.predict(image)

    pred = np.argmax(pred)

    # pred = labels[pred]

    return render_template("prediction.html", data=pred)


if __name__ == "__main__":
    app.run(debug=True)
