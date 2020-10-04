import time
from yolov3_tf2.utils import glo
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs

from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton,  QPlainTextEdit,QMessageBox
from PySide2.QtUiTools import QUiLoader
import sys
import socket
from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
signal(SIGPIPE, SIG_IGN)
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', '0','0')
#flags.DEFINE_string('video', './data/video.mp4',
 #                   'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    test1 = ui.key1_2.text()
    test2 = ui.key2_2.text()
    test3 = ui.key3_2.text()
    test4 = ui.key4_2.text()
    test5 = ui.key5_2.text()
    test6 = ui.key2_1.text()
    test7 = ui.key2_3.text()
    test8 = ui.key2_4.text()
    test9 = ui.key2_5.text()
    test10 = ui.key2_6.text()
    test11 = ui.key3_1.text()
    test12 = ui.key3_3.text()
    test13 = ui.key3_4.text()
    test14 = ui.key3_5.text()
    test15 = ui.key3_6.text()
    
    ui.progressBar.setValue(1)
    global box1,box2
    #box1,box2 = 1,1
    #ami = ui.comboBox.currentText()
    #print(ami)
    #mo = ui.box2.currentText()
    #print(mo)
    #if ami == 'open':
        #box1 = 1
    #else:
	#box1 = 0
    #if mo == 'open':
    #	box2 = 1
    #else:
	#box2 = 0

    ip_ = ui.IpAdr.text()
    port = ui.lineEdit_2.text()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
    s.bind((ip_,int(port)))
    #s.bind(('192.168.43.99',8000))
    s.listen(5)
    msg,addr = s.accept()
    QMessageBox.information(ui,'Welcome','Game start!')
        
    ui.progressBar.setValue(2)
	
    #print(msg.rev(1024).decode('utf-8'))
    #glo.port = ui.lineEdit_2.text()
    #print("******************")
    #print(glo.port)
    #print("******************")
    #print(test1)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')


    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]
	
	ui.progressBar.setValue(3)
	    
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names,test1,test2,test3,test4,test5,test6,test7,test8,test9,test10,test11,test12,test13,test14,test15,msg,box1,box2)
        img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        if FLAGS.output:
            out.write(img)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

def run():
    #QMessageBox.information(ui,'Welcome','Game start!')
    try:
        app.run(main)
    except SystemExit:
        pass

def stop():
    sys.exit()

def IPAdress():
    ipAdress = ui.IpAdr.text()
    port = ui.lineEdit_2.text()

def change1():
    global box1
    ami = ui.comboBox.currentText()
    if ami == 'open':
        box1 = 1
    else:
	box1 = 0
    
def change2():
    global box2
    ji = ui.box2.currentText()
    if ji == 'open':
	box2 = 1
    else:
	box2 = 0
    

app1 = QApplication([])
ui = QUiLoader().load('./QT/PC_controller.ui')
global box1,box2
box1 = 1
box2 = 1
ui.comboBox.currentIndexChanged.connect(change1)
ui.box2.currentIndexChanged.connect(change2)
ui.progressBar.setRange(0,3)
ui.button_run.clicked.connect(run)
ui.button_stop.clicked.connect(stop)


ui.show()
app1.exec_()


