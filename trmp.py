# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 19:22:50 2019

@author: czz
"""
from gluoncv import data, utils
from mxnet import gluon

import mxnet as mx
import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths
from imutils.video import FileVideoStream
import numpy as np
import os, imutils
#gd.flv
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
classes = ['hat', 'person']
ctx = mx.cpu()
#
#
# frame = '1.jpg'
# img = cv2.imread(frame)

# x, img = data.transforms.presets.yolo.load_test(frame, short=416)
# x = x.as_in_context(ctx)
# net=gluon.SymbolBlock.imports(symbol_file='./darknet53-symbol.json', input_names=['data'], param_file='./darknet53-0000.params', ctx=ctx)
#
#
#
# box_ids, scores, bboxes = net(x)
# ax = utils.viz.cv_plot_bbox(img, bboxes[0], scores[0], box_ids[0], class_names=classes,thresh=0.4)
# cv2.imshow('image', img[...,::-1])
# cv2.waitKey(0)
# cv2.imwrite(frame.split('.')[0] + '_result.jpg', img[...,::-1])
# cv2.destroyAllWindows()


# cap = cv2.VideoCapture("rtmp://58.200.131.2:1935/livetv/hunantv")
cap = cv2.VideoCapture("rtmp://58.200.131.2:1935/livetv/cctv7")

ret,frame = cap.read()
frame_index = 0
while ret:
    ret,frame = cap.read()
    if frame_index%12==0:
        newpath = os.path.join(r'image/capture/', str(frame_index) + ".jpg");
        cv2.imwrite( newpath,frame)
        img = newpath
        x, frame = data.transforms.presets.yolo.load_test(img, short=416)
        x = x.as_in_context(ctx)
        net=gluon.SymbolBlock.imports(symbol_file='./darknet53-symbol.json', input_names=['data'], param_file='./darknet53-0000.params', ctx=ctx)
        box_ids, scores, bboxes = net(x)

        if isinstance(bboxes, mx.nd.NDArray):
            bboxes = bboxes.asnumpy()

        if isinstance(box_ids, mx.nd.NDArray):
            box_ids = box_ids.asnumpy()

        if isinstance(scores, mx.nd.NDArray):
            scores = scores.asnumpy()

        length = scores[0]
        # 获得1帧内预测个数
        num = 0;
        for i in range(0, len(length)):
            if scores[0][i] != -1:
                num += 1
            else:
                break
        for j in range(0, num):
            print(box_ids[0][j])
            print(scores[0][j])
            print(bboxes[0][j])


        ax = utils.viz.cv_plot_bbox(frame, bboxes[0], scores[0], box_ids[0], class_names=classes, thresh=0.4)

        flag = 1
        if flag == 0:
            os.remove(newpath)
        else:
            cv2.imwrite(newpath,frame)

        cv2.imshow("frame", frame)
    frame_index += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
