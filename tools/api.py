from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = load_model('trained_plant_disease_model7.h5')
# model = load_model('trained_plant_disease_model7.keras')

# Dictionary of class names
class_names = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___healthy',
    37: 'not_plant_leaf(please try agin)'
}


# Function to preprocess the image
def preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Resize image to match model's expected sizing
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


# Function to make predictions
def predict_disease(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    if img is None:
        return None
    # Make predictions
    predictions = model.predict(img)
    # Get the predicted class label
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]


# Define a route to handle image requests
@app.route('/predict', methods=['POST'])
def predict():
    # Receive the image data
    image_data = request.files['image']
    # Pass the image data to your model for prediction
    # Replace this with your actual model prediction code
    prediction = predict_with_model(image_data)
    # Return the prediction result as a response
    return jsonify({'prediction': prediction})


# Define a function to perform model prediction
def predict_with_model(image_data):
    # image_path = 'uploaded_image.jpg'
    # image_data.save(image_path)
    # Example usage
    result_index = predict_disease(image_data)
    if result_index is not None:
        print("Predicted class:", class_names.get(result_index, "Unknown"))
        return class_names.get(result_index, "Unknown")


if __name__ == '__main__':
    app.run(debug=True)
