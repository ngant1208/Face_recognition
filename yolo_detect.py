#########detect face############
from numpy import expand_dims,asarray
# from keras.models import load_model
# from keras.utils.image_utils import load_img
from keras.utils.image_utils import img_to_array
import cv2
import numpy as np
from keras import backend as K
# from tensorflow.keras import backend as K
# from matplotlib import pyplot
# import os
# import argparse
# import sys
from PIL import Image

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2] # 0 and 1 is row and column 13*13
    nb_box = 3 # 3 anchor boxes
    netout = netout.reshape((grid_h, grid_w, nb_box, -1)) #13*13*3 ,-1
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh
    
    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if(objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

#intersection over union        
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin  
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union

def do_nms(boxes, nms_thresh):    #boxes from correct_yolo_boxes and  decode_netout
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


# load and prepare an image
def load_image_pixels(filename, shape):
    image = Image.open(filename)
    width, height = image.size
    image = image.convert('RGB')
    pixels = asarray(image)
    image = image.resize(shape)
    image=asarray(image)
    image = image.astype('float32')
    image /= 255.0  #rescale the pixel values from 0-255 to 0-1 32-bit floating point values.
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height,pixels
# get all of the results above a threshold
def get_boxes(boxes,labels,thresh):
    v_boxes = list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
    return v_boxes
    
def detect_face_image(file_name,model,labels=["face"]):
    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
    shape=(416,416)
    image, image_w, image_h,pixels = load_image_pixels(file_name,shape)
    yhat = model.predict(image)
    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], 0.6, image_w,image_h)
        
    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, image_w,image_h)

    # suppress non-maximal boxes
    do_nms(boxes, 0.5)  #Discard all boxes with pc less or equal to 0.5

    # get the details of the detected objects
    v_boxes = get_boxes(boxes, labels, 0.6)
    y1=0
    y2=0
    x1=0
    x2=0
    if len(v_boxes)>0:
        for box in v_boxes:
            y1=box.ymin
            y2=box.ymax
            x1=box.xmin
            x2=box.xmax
    face = pixels[y1:y2, x1:x2]
    K.clear_session() 
    return face
#for videostream
# load and prepare an image
def load_image_pixels_frame(frame, shape):
    image=cv2.resize(frame,shape)
    image = img_to_array(image)
	# scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
	# add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image

def detect_face_frame(yolo_model,frame,image_w,image_h,input_w,input_h,anchors,labels=["face"],class_threshold=0.6):
    image = load_image_pixels_frame(frame, (input_w, input_h))
    yhat = yolo_model.predict(image)
    # summarize the shape of the list of arrays
    # print([a.shape for a in yhat])
    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
        # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    # suppress non-maximal boxes
    do_nms(boxes, 0.5)
    # define the labels
    labels = ["face"]
    # get the details of the detected objects
    v_boxes= get_boxes(boxes, labels, class_threshold)
    faces=list()
    #fix for facenet
    y1=0
    y2=0
    x1=0
    x2=0
    for box in v_boxes:
        y1=box.ymin
        y2=box.ymax
        x1=box.xmin
        x2=box.xmax
        face=frame[y1:y2, x1:x2]
        # faces+=face
    K.clear_session() 
    return face,y1,y2,x1,x2




# def detect_face(photo_filename):
#     anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]  
#     class_threshold = 0.6
#     #define class
#     labels = ["face"]
#     input_w, input_h = 416, 416
#     # load and prepare image
#     image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))

#     # load yolov3 model
#     model = load_model('yolo_model/model.h5',compile=False)
#     yhat = model.predict(image)
#     # summarize the shape of the list of arrays
#     print([a.shape for a in yhat])
#     boxes = list()
#     for i in range(len(yhat)):
#         # decode the output of the network
#         boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
        
#     # correct the sizes of the bounding boxes for the shape of the image
#     correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

#     # suppress non-maximal boxes
#     do_nms(boxes, 0.5)  #Discard all boxes with pc less or equal to 0.5

#     # get the details of the detected objects
#     v_boxes = get_boxes(boxes, labels, class_threshold)
#     K.clear_session() 
#     return v_boxes

# if __name__ == "__main__":
#     _main()










