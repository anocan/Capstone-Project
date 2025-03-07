import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model("models/MODEL-X.keras")

# Labels for the classes
names = ["breasts", "butterfliedDrumsticks", "drumsticks", "wholeLeg", "wings"]

t_target_x = 224
t_target_y = 224

# Function to preprocess a single image
def preprocess_image(image_path):
    # Load the image with target size
    img = load_img(image_path, target_size=(t_target_x, t_target_y))
    # Convert the image to an array
    img_array = img_to_array(img)
    # Expand dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image (MobileNetV2 preprocess)
    return preprocess_input(img_array)

#"testData"
PATH = "testData/"


# Path to a new image
image_path = "testmodule/example_set1/butterflyDrumsticks10foot/ex5.jpg"   # Replace with the path to your image

# Preprocess the image
input_data = preprocess_image(image_path)

# Make a prediction
predictions = model.predict(input_data)

print(predictions)

# Decode the predictions
predicted_label = np.argmax(predictions, axis=1)
confidence = np.max(predictions)

# Display the result
print(f"Predicted Label: {names[predicted_label[0]]}")
print(f"Confidence: {confidence:.2f}")

# Visualize the input image with the prediction
img = load_img(image_path)
plt.imshow(img)
plt.title(f"Prediction: {names[predicted_label[0]]} ({confidence:.2f})")
plt.axis('off')
plt.show()
