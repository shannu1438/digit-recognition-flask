from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model("digit_model.h5")  # Load trained model

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No file uploaded.")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected.")

    # Preprocess the uploaded image
    image = Image.open(file).convert('L')       # Convert to grayscale
    image = image.resize((28, 28))              # Resize to 28x28
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0                       # Normalize

    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    return render_template('index.html', prediction=predicted_digit)

if __name__ == '__main__':
    print("Flask app starting...")
    app.run( debug=True )
