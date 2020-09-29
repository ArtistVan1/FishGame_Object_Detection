import socket
import time
from PySide2.QtUiTools import QUiLoader
soc = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
HOST = '192.168.43.99'
PORT = 8000
soc.connect((HOST,PORT))
#x_location = soc.recv(1024).decode('utf-8')


output = 'A'
output_old = 'A'
while True:
	output = soc.recv(1).decode('utf-8')
	if output != output_old:
		print(output)
	output_old = output
	
