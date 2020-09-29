from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5 import uic
import RPi.GPIO as GPIO
import socket
import sys
from threading import Thread
import  inspect
import  ctypes

def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")
def run():
    global thread
    print("program run")
    HOST = ui.ipAdress.test()
    PORT = ui.port.test()
    soc = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    soc.connect((HOST,PORT))
    thread = Thread(target=newThreadFunc,args=(soc))
    thread.start()

def newThreadFunc(soc):
    global output,output_old
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(18,GPIO.OUT)
    GPIO.setup(24,GPIO.OUT)
    GPIO.setup(29,GPIO.OUT)
    GPIO.setup(31, GPIO.OUT)
    GPIO.setup(32, GPIO.OUT)
    GPIO.setup(33, GPIO.OUT)
    GPIO.setup(35, GPIO.OUT)
    GPIO.setup(36, GPIO.OUT)
    GPIO.setup(37, GPIO.OUT)
    GPIO.setup(38, GPIO.OUT)
    GPIO.setup(40, GPIO.OUT)
    pwm = GPIO.PWM(18,1)
    pwm.start(50)
    dic = {'up':24,'down':29,'left':31,'right':32,'A':33,'B':35,'Y':36,'X':37,'L':38,'R':40}

    output = soc.recv(1).decode('utf-8')
    if output != output_old:
        gpio_ = dic[output]
        pwm.stop()
        pwm = GPIO.PWM(int(gpio_),1)
        pwm.start(50)
        print(output,'  ',gpio_)
    output_old = output

def stop():
    global thread
    print("game stop")
    _async_raise(thread.ident,SystemExit)
    sys.exit()

global thread
global output,output_old
output,output_old = 'A','A'
app = QApplication([])
ui = uic.loadUi('UI/RS_controller.ui')

ui.run.clicked.connect(run)
ui.stop.clicked.connect(stop)
ui.show()
app.exec_()
