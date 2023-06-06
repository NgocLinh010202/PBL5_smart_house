import cv2
import io
import socket
import struct
import pickle

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('10.2.6.55', 5757))

soc_receive_name = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
soc_receive_name.bind(('10.2.6.55', 9000))
soc_receive_name.listen(10)
print('Socket receive name now listening')
print('Socket now listening')



connection = client_socket.makefile('wb')

# cam = cv2.VideoCapture('./test_suong.mp4')
cam = cv2.VideoCapture(0)
cam.set(3, 320)
cam.set(4, 240)
img_counter = 0
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

check_soc_receive_name_connected = False

while True:
    ret, frame = cam.read()
    result, frame = cv2.imencode('.jpg', frame, encode_param)
    data = pickle.dumps(frame, 0)
    size = len(data)

    print("{}: {}".format(img_counter, size))
    client_socket.sendall(struct.pack(">L", size) + data)
    img_counter += 1
    
    if check_soc_receive_name_connected == False:
        conn, addr = soc_receive_name.accept()
        check_soc_receive_name_connected = True
    print('conn: ', conn)
    print('addr: ', addr)

    dataFromServer = conn.recv(1024)
    if dataFromServer:
        name = dataFromServer.decode()
        print('Received:', name)
    else:
        print('No received')
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
