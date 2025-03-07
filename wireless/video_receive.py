import cv2
import socket
import struct
import pickle

import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import img_to_array # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
import tensorflow as tf

# Set up the socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", 9999))
server_socket.listen(5)

print("Waiting for connection...")
client_socket, addr = server_socket.accept()
print(f"Connected to {addr}")

data = b""
payload_size = struct.calcsize("Q")

model = load_model("models/MODEL-X.keras")
# Class names
names = ["breasts", "butterfliedDrumsticks", "drumsticks", "wholeLeg", "wings"]

# Preprocess function (consistent with ResNet50)
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Resize frame to model input size
    frame_array = img_to_array(frame_resized)
    frame_array = np.expand_dims(frame_array, axis=0)
    return preprocess_input(frame_array)

def get_crop_region(frame_width, frame_height):
    rect_width = int(frame_width * 0.8)
    rect_height = int(frame_height * 0.9)
    rect_x1 = (frame_width - rect_width) // 2
    rect_y1 = (frame_height - rect_height) // 2
    return rect_x1, rect_y1, rect_width, rect_height


while True:
    while len(data) < payload_size:
        packet = client_socket.recv(4096)
        if not packet:
            break
        data += packet

    if not data:
        break

    packed_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_size)[0]

    while len(data) < msg_size:
        data += client_socket.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Deserialize frame
    frame = pickle.loads(frame_data)

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
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()
server_socket.close()
cv2.destroyAllWindows()
