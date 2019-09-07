import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization
import threading


# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)
# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
#data_format = 'NCHW'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
'''
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

'''
ckpt = tf.train.get_checkpoint_state('../train_model')
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt.model_checkpoint_path)
# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# Main image processing routine.
def process_image(img, select_threshold=0.35, nms_threshold=.01, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

def paint_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    img_painted = img
    height = img_painted.shape[0]
    width = img_painted.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            print(score)
            if cls_id not in colors :
                colors[cls_id] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                print(colors[cls_id])
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            cv2.rectangle(img_painted, (xmin,ymin), (xmax, ymax), colors[cls_id], 1)
            class_name = str(cls_id)
            cv2.putText(img_painted, '{:s} | {:.3f}'.format(class_name, score), (xmin,ymin), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[cls_id], 2, 1)
    return img_painted

'''
# Test on some demo image and visualize output.
# 测试的文件夹
path = '../demo/'
image_names = sorted(os.listdir(path))
# 文件夹中的第几张图，-1代表最后一张
img = mpimg.imread(path + image_names[-2])
rclasses, rscores, rbboxes = process_image(img)

# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

'''
'''
class A():
    aa = ""


class tt(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while True:
            A.aa = input('enter:')
            if A.aa == 'Q':
                break

def main():
    my_t = tt()
    my_t.start()
    path = '../demo/test/'
    image_names = sorted(os.listdir(path))
    num = len(image_names)
    print(num)
    i = -1
    while True:
        if A.aa == "A":
            continue
        elif A.aa == "Q":
            break
        else:
            i = i + 1
            if i == num:
                break
            else:
                print(i)
                img = cv2.imread(path + image_names[i], -1)
                rclasses, rscores, rbboxes = process_image(img)
                img_painted = paint_bboxes(img, rclasses, rscores, rbboxes)
                cv2.imshow('Figure 1', img_painted)
                cv2.waitKey(5)


main()
'''

def paint_rect(img, classes, scores, bboxes):
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            cv2.rectangle(img, (ymin, xmin), (ymax, xmax), colors[cls_id], 1)
    return img


def main(camera):
    if camera == True:
        cv2.namedWindow("camera", 1)
        #cv2.namedWindow("camera", 1)
        #video = "http://admin:admin@10.26.253.250:8081/"  # 此处@后的ipv4 地址需要修改为自己的地址
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 1)
        while True:
            # get a frame
            ret, frame = cap.read()
            # show a frame
            #if cv2.waitKey(1) & 0xFF == ord('a'):
            rclasses, rscores, rbboxes = process_image(frame)
            img = paint_bboxes(frame,rclasses, rscores, rbboxes)
            cv2.imshow('capture',img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()




if __name__ == '__main__':
    main(True)
