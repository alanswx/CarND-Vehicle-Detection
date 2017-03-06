import _init_paths
import tensorflow as tf
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network
import lanefind

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
def draw_bboxes(img, bboxes,box_color=(255,0,0)):
    for bbox in bboxes:
        #print(bbox)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 6)

def vis_detections_draw(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        #cv2.rectangle(im, bbox[0], bbox[1], (255,244,0), 6)
        if (class_name!='boat'):
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,244,0), 4)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = '{:s} {:.3f}'.format(class_name, score)
            cv2.putText(im,text,(int(bbox[0]), int(bbox[1]) - 2), font, 1,(255,255,255),2)

    return im

  
def labelFrame(sess, net, im):
    im=cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    scores, boxes = im_detect(sess, net, im)
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        
        #draw_bboxes(im, cls_boxes,box_color=(255,0,0))

        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections_draw(im, cls, dets, thresh=CONF_THRESH)
    im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im  

        

cfg.TEST.HAS_RPN = True  # Use RPN for proposals

# init session
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
# load network
net = get_network("VGGnet_test")
    
# load model
saver = tf.train.Saver()
model_name = './VGGnet_fast_rcnn_iter_70000.ckpt'
saver.restore(sess, model_name)
    #sess.run(tf.initialize_all_variables())

print ('\n\nLoaded network {:s}'.format(model_name))
from moviepy.editor import VideoFileClip
def pipeLine(image):
    image = lanefind.lanePipeline(image)
    image=labelFrame(sess, net, image)
    return image

def processCombinedVideo(input_video,output):
    clip1 = VideoFileClip(input_video)
    out_clip = clip1.fl_image(pipeLine)
    out_clip.write_videofile(output,audio=False) 

#processCombinedVideo('test_video.mp4','test_video_faster-rcnn-out.mp4')
#processCombinedVideo('project_video.mp4','project_video_faster-rcnn-out.mp4')
processCombinedVideo('bayshore.mp4','bayshore_faster-rcnn-out.mp4')
