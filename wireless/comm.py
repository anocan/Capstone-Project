import socket

HOST = '192.168.137.125'  # Replace with Raspberry Pi's IP address
PORT = 12345          # Same port as server

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

message = "Hello from PC!"
client_socket.sendall(message.encode())

response = client_socket.recv(1024)
print("Server response:", response.decode())

client_socket.close()
