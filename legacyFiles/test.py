from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

### LEGACY !!!!

PATH = "testData/"

# Load the trained model
model = load_model("chicken_with_mobilenetv2.keras")
model.summary()

# Class names
names = ["breasts", "butterfliedDrumsticks", "drumsticks", "wholeLeg", "wings"]

# Preprocess the image
image_path = f"{PATH}test01.jpg" 
img = load_img(image_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict the class
predictions = model.predict(img_array)
predictions = tf.nn.softmax(predictions)
predicted_class = np.argmax(predictions, axis=1)[0]
predicted_label = names[predicted_class]

print("Class probabilities:")
for i, class_name in enumerate(names):
    confidence = predictions[0][i] * 100  # Convert to percentage
    print(f"{class_name}: {confidence:.2f}%")

print(f"Predicted Class: {predicted_label}")

# Visualize
plt.imshow(img)
plt.title(f"Predicted: {predicted_label}")
plt.axis("off")
plt.show()
