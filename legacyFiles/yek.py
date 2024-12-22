import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import tensorflow as tf

### LEGACY !!!!

# Load the trained model
model = load_model("chicken_with_mobilenetv2-P.keras")

# Class names
names = ["breasts", "butterfliedDrumsticks", "drumsticks", "wholeLeg", "wings"]

# Video file path (pre-recorded video)
VIDEO_PATH = "/Users/anilbudak/VSCode/Bitirme/CNN/testData/Parameter Input Single Type/Butterflied Drumsticks/IMG_8679.MOV"  # Update this with your video file path

#testData/FINAL/v2.MOV

# Preprocess function (similar to image preprocessing)
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Resize frame to model input size
    frame_array = preprocess_input(img_to_array(frame_resized))
    return np.expand_dims(frame_array, axis=0)

t_target_x = 224
t_target_y = 224


# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# (x, y, width, height)
crop_region = (200, int(1080/2), int(1080*0.7), int(1920*0.4))

while True:
    # Read frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("End of video file.")
        break

    # Extract the cropped section
    x, y, w, h = crop_region
    cropped_frame = frame[y:y+h, x:x+w]

    # Preprocess the cropped frame
    processed_frame = preprocess_frame(cropped_frame)

    # Predict the class
    predictions = model.predict(processed_frame)
    predictions = tf.nn.softmax(predictions)
    print(predictions)
    #predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)


    # Display prediction on the frame
    #confidence = predictions[0][predicted_class] * 100
    if confidence < 0.2:
        label_text = "Unsatisfactory Confidance <0.4"
    else:
        label_text = f"{names[predicted_label[0]]}: {confidence:.2f}%"
    cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw a rectangle to indicate the cropped region
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the full frame with the highlighted cropped section
    cv2.imshow("Chicken Product Classification", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and close windows
cap.release()
cv2.destroyAllWindows()
