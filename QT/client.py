import socket
import time
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(2,GPIO.OUT)
pwm = GPIO.PWM(2,1) #1Hz
pwm.start(50)
#from PySide2.QtUiTools import QUiLoader
soc = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
HOST = '192.168.43.99'
PORT = 8001
soc.connect((HOST,PORT))
#x_location = soc.recv(1024).decode('utf-8')

dic = {'up':24,'down':29,'left':31,'right':32,'A':33,'B':35,'Y':36,'X':37,'L':38,'R':40}
output = 'A'
output_old = 'A'
while True:
	output = soc.recv(1).decode('utf-8')
	if output != output_old:
		gpio_ = dic[output]
		print(output,'  ', gpio_)
		
		pwm.stop()
		GPIO.setup(gpio_,GPIO.OUT)
		pwm = GPIO.PWM(gpio_,1)
		pwm.start(50)
	output_old = output
	
