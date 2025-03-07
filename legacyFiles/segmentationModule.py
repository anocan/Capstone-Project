import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from legacyFiles.segmentation import segment_frame  # Import your segmentation function

# Load the trained model
model = load_model("models/MODEL-X.keras")

# Class names
names = ["breasts", "butterfliedDrumsticks", "drumsticks", "wholeLeg", "wings"]

# Video file path (pre-recorded video)
VIDEO_PATH = "testData/videos/Custom/Parameter-Input-Single-Type/drumsticks/IMG_8666.MOV"  # Update this with your video file path

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

# Main processing loop
while True:
    # Read frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("End of video file.")
        break

    # Segment the frame to isolate the product
    masked_frame, bounding_box = segment_frame(frame)

    if masked_frame is not None:
        # Preprocess the segmented (masked) frame for prediction
        processed_frame = preprocess_frame(masked_frame)

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

        # Draw the bounding box on the original frame
        cv2.drawContours(frame, [bounding_box], -1, (0, 255, 0), 2)
        cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        # If no segmentation, display a message
        label_text = "No valid product detected"
        cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the frame with prediction and bounding box
    cv2.imshow("Chicken Product Classification", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and close windows
cap.release()
cv2.destroyAllWindows()
