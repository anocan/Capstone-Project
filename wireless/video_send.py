import cv2
import socket
import struct
import pickle

# Set up the socket
server_ip = "192.168.137.83"  # Replace with the receiver laptop's IP address
server_port = 9999

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))

# Open laptop camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Serialize the frame
    data = pickle.dumps(frame)
    size = struct.pack("Q", len(data))

    # Send size first, then data
    client_socket.sendall(size + data)

    # Display sent frame (optional)
    cv2.imshow("Sender", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
client_socket.close()
cv2.destroyAllWindows()
