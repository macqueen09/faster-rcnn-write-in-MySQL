#!/usr/bin/env python
# -*- coding: utf-8 -*-

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

import math
import MySQLdb as mdb
import time

CLASSES = ('__background__',
           'tikang','burke','perry','freedom','dependence','cv','car','person','aeroplane','bus','train','bicycle')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        '12ship': ('ZF',
                  'zf_faster_rcnn_iter_70000.caffemodel')}

NN = 6
def vis_detections(im, class_name, dets,file_dir,FrameID, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return 0

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    shipid_now = get_vid()-1
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

        ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
        print "this class is: ",class_name,"the score:",score,"box is ",bbox
        ISOTIMEFORMAT='%Y-%m-%d %X'
        time_det = time.strftime(ISOTIMEFORMAT,time.localtime(time.time() ) )
        print "time is ",str(time_det)
        insertsql((shipid_now+1),FrameID,int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),class_name,score,file_dir,str(time_det))
    # plt.axis('off')
    # plt.tight_layout()
    # plt.draw()

    return len(inds)

def demo(net,image_name,file_dir,FrameID):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(file_dir, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    time_det = timer.total_time

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    all_obj = 0
    for cls_ind, cls in enumerate(CLASSES[1:NN+1]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        tempallname = os.path.join(file_dir,image_name)
        num_of_ship = vis_detections(im, cls, dets,tempallname,FrameID, thresh=CONF_THRESH)#im, cls, dets, thresh=CONF_THRESH
        all_obj += num_of_ship
    return all_obj

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='12ship')
    args = parser.parse_args()
    return args

def get_vid():
    #try:
    conn = mdb.connect(host="localhost",user="root",passwd="root",db="test")
    query='SELECT shipID FROM ship_detection ORDER BY shipID DESC LIMIT 1;'
    cur=conn.cursor()
    cur.execute(query)
    row = cur.fetchone()
    if row:
        vid = row[0] +1
    else:
        vid = 1
    cur.close()
    conn.commit()
    conn.close()
    return vid

def insertsql(shipID=0, im_Frame=4, x1=0, y1=0, x2=0, y2=0, im_category=3, scoree=0.56, im_path="path__path", im_time="time0.98"):
    conn = mdb.connect(host="localhost",user="root",passwd="root",db="test")
    cursor = conn.cursor()
    # get the number of obj
    cursor.execute("select * from ship_detection")
    rows = cursor.fetchall()
    num_pic = len(rows)

    cursor.execute("SELECT VERSION()")
    # 
    sql = ("insert into ship_detection values({:d},{:d},{:f},{:f},{:f},{:f},\"{:s}\",\"{:s}\",\"{:s}\",\"{:s}\")").\
    format(shipID,im_Frame,float(x1),float(y1),float(x2),float(y2),str(im_category),str(scoree),str(im_path),str(im_time))
    print "sql command is ",sql
    try:
        cursor.execute(sql)
        conn.commit()
        print "write sql success"
    except mdb.Error, e:
        # Rollback in case there is any error
        conn.rollback()
        print "MySQL Error %d: %s" % (e.args[0], e.args[1])
    conn.close()


def main_detect(mac_img,FrameID):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    print "pic name: ",mac_img
    im_name = os.path.basename(mac_img)
    print "im_name : ",im_name
    file_dir = os.path.dirname(mac_img)
    print "file_dir : ",file_dir
    print 'Demo for img_{}~~~~~~~~~~~~~~~~~~~'.format(mac_img)
    all_ship = demo(net,im_name,file_dir,FrameID)

    print "there are ships :",all_ship
    # plt.show()
    return all_ship

