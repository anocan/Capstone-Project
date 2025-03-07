import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import img_to_array # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
import tensorflow as tf

# Load the trained model
model = load_model("models/MODEL-X.keras")

# Class names
names = ["breasts", "butterfliedDrumsticks", "drumsticks", "wholeLeg", "wings"]

# Video file path (pre-recorded video)
VIDEO_PATH = "testData/videos/Custom/Parameter-Input-Frequency/2x/IMG_8658_2x.mp4"  # Update this with your video file path

# Preprocess function (consistent with ResNet50)
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Resize frame to model input size
    frame_array = img_to_array(frame_resized)
    frame_array = np.expand_dims(frame_array, axis=0)
    return preprocess_input(frame_array)

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Define cropping region (central 50% of the frame)
def get_crop_region(frame_width, frame_height):
    rect_width = int(frame_width * 0.5)
    rect_height = int(frame_height * 0.6)
    rect_x1 = (frame_width - rect_width) // 2
    rect_y1 = (frame_height - rect_height) // 2
    return rect_x1, rect_y1, rect_width, rect_height

# Main processing loop
while True:
    # Read frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("End of video file.")
        break

    # Get frame dimensions and define crop region
    frame_height, frame_width = frame.shape[:2]
    x, y, w, h = get_crop_region(frame_width, frame_height)

    # Extract and preprocess the cropped section
    cropped_frame = frame[y:y+h, x:x+w]
    processed_frame = preprocess_frame(cropped_frame)

    # Predict the class
    predictions = model.predict(processed_frame)
    predicted_label = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)

    # Display prediction only if confidence is high enough
    confidence_threshold = 0.5
    if confidence >= confidence_threshold:
        label_text = f"{names[predicted_label[0]]}: {confidence:.2f}"
    else:
        label_text = "No product detected"

    # Draw a rectangle to indicate the cropped region
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the full frame with the highlighted cropped section
    cv2.imshow("Chicken Product Classification", frame)

    # Press 'q' to exit

    key = cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord(' '):  # Press spacebar to advance to the next frame
        continue

    """
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    """


# Release video and close windows
cap.release()
cv2.destroyAllWindows()
