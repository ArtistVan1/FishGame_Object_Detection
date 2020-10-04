from absl import logging
import numpy as np
import tensorflow as tf
import cv2
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton,  QPlainTextEdit
from PySide2.QtUiTools import QUiLoader
import glo
from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
signal(SIGPIPE, SIG_IGN)

output_old = 'A'

print("Port:{}".format(glo.port))

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)
def jude(x,y,test1,test2,test3,test4,test5,test6,test7,test8,test9,test10,test11,test12,test13,test14,test15):
    x = int(x)
    y = int(y)
    output = 'A'
    if x<128:
        if y<160:
	    output = test1
	if 160<y<320:
	    output = test6
	if 320<y<480:
	    output = test11
    if 128<x<256:
	if y<160:
       	    output = test2
	if 160<y<320:
	    output = test7
	if 320<y<480:
	    output = test12
    if 256<x<384:
	if y<160:
	    output = test3
	if 160<y<320:
	    output = test8
	if 320<y<480:
	    output = test13
    if 384<x<512:
	if y<160:
	    output = test4
	if 160<y<320:
	    output = test9
	if 320<y<480:
	    output = test14
    if 512<x<640:
	if y<160:
            output = test5
	if 160<y<320:
	    output = test10
	if 320<y<480:
	    output = test15
   # print(output)
    return output

def draw_outputs(img, outputs, class_names,test1,test2,test3,test4,test5,test6,test7,test8,test9,test10,test11,test12,test13,test14,test15,msg):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
	x1 = x1y1[0]
	y1 = x1y1[1]
	x2 = x2y2[0]
	y2 = x2y2[1]
	center_x = (x1+x2)/2
	center_y = (y1+y2)/2
	center = (center_x,center_y)
 	img = cv2.line(img,(0,160),(640,160),(0,0,255),3,4)
	img = cv2.line(img,(0,320),(640,320),(0,0,255),3,4)
	img = cv2.line(img,(128,0),(128,480),(0,0,255),3,4)
	img = cv2.line(img,(256,0),(256,480),(0,0,255),3,4)
	img = cv2.line(img,(384,0),(384,480),(0,0,255),3,4)
	img = cv2.line(img,(512,0),(512,480),(0,0,255),3,4)
	
	img = cv2.putText(img,test1,(80,64),cv2.FONT_HERSHEY_COMPLEX_SMALL,2.0,(100,200,200),5)
	img = cv2.putText(img,test2,(192,64),cv2.FONT_HERSHEY_COMPLEX_SMALL,2.0,(100,200,200),5)
	img = cv2.putText(img,test3,(320,64),cv2.FONT_HERSHEY_COMPLEX_SMALL,2.0,(100,200,200),5)
   	img = cv2.putText(img,test4,(448,64),cv2.FONT_HERSHEY_COMPLEX_SMALL,2.0,		(100,200,200),5)
	img = cv2.putText(img,test5,(576,64),cv2.FONT_HERSHEY_COMPLEX_SMALL,2.0,(100,200,200),5)
	img = cv2.putText(img,test6,(64,240),cv2.FONT_HERSHEY_COMPLEX_SMALL,2.0,(100,200,200),5)
	img = cv2.putText(img,test7,(192,240),cv2.FONT_HERSHEY_COMPLEX_SMALL,2.0,(100,200,200),5)
	img = cv2.putText(img,test8,(320,240),cv2.FONT_HERSHEY_COMPLEX_SMALL,2.0,(100,200,200),5)
	img = cv2.putText(img,test9,(448,240),cv2.FONT_HERSHEY_COMPLEX_SMALL,2.0,(100,200,200),5)
	img = cv2.putText(img,test10,(576,240),cv2.FONT_HERSHEY_COMPLEX_SMALL,2.0,(100,200,200),5)
	img = cv2.putText(img,test11,(64,400),cv2.FONT_HERSHEY_COMPLEX_SMALL,2.0,(100,200,200),5)
	img = cv2.putText(img,test12,(192,400),cv2.FONT_HERSHEY_COMPLEX_SMALL,2.0,(100,200,200),5)
	img = cv2.putText(img,test13,(320,400),cv2.FONT_HERSHEY_COMPLEX_SMALL,2.0,(100,200,200),5)
	img = cv2.putText(img,test14,(448,400),cv2.FONT_HERSHEY_COMPLEX_SMALL,2.0,(100,200,200),5)
	img = cv2.putText(img,test15,(576,400),cv2.FONT_HERSHEY_COMPLEX_SMALL,2.0,(100,200,200),5)
	if class_names[int(classes[i])]=='person':
            #img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
	    img = cv2.circle(img,center,5,(0,0,255),4)
	    output = jude(center_x,center_y,test1,test2,test3,test4,test5,test6,test7,test8,test9,test10,test11,test12,test13,test14,test15)
	    signal(SIGPIPE, SIG_IGN)
	    msg.send(str(output).encode())
	    print(output)
	    #msg.send(str(center_y).encode())
	#print(str(center))
	#print(class_names[int(classes[i])])
	#if class_names[int(classes[i])]=='person': 
	#print(img.shape)
            img = cv2.putText(img, '{} {:.4f}'.format(
                class_names[int(classes[i])], objectness[i]),
                x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


def draw_labels(x, y, class_names):
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, class_names[classes[i]],
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 255), 2)
    return img


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)

