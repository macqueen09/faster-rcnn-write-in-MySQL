#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

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
import subprocess

from ship_detect import insertsql
from ship_detect import get_vid
import time
import math
import MySQLdb as mdb

CLASSES = ('__background__',
            'tikang','burke','perry','freedom','dependence','cv','car','person','aeroplane','bus','train','bicycle')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        '12ship':('ZF',
                  'zf_faster_rcnn_iter_70000.caffemodel')}
NN=6
all_flog = 1
# no use
# output:
#   rectresults: [rect1,rect2,rect3,rect4,score, classname]
#   imgresultpath
def plotresults(im, class_name, dets, image_name, results_dir,thresh):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return -1

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    rectresults = []
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        temp = list(dets[i,:])
        temp.append(class_name)
        rectresults.append(temp)
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
    plt.axis('off')
    plt.tight_layout()
    imageresultpath =  os.path.join(results_dir,"res_"+  image_name + "_"+ class_name+'.png')
    plt.savefig(imageresultpath)
    # print rectresults
    return rectresults,imageresultpath

# input: net, imgpath, imgname, resultdir,
# output 
# rectresults: [[5.5272789, 94.176041, 126.43796, 203.25676, 0.80156678, 'car'], [256.823, 15.384331, 354.41022, 224.1442, 0.99822456, 'person'], [430.33359, 123.24626, 451.40884, 186.96574, 0.98412746, 'person']]  
# imageresultpath: each for one image, e.g /home/fenglei/project/hjj/detection/py-faster-rcnn/data/results/res_004545.jpg_person.png
# 
def get_FrameID():
    conn = mdb.connect(host="localhost",user="root",passwd="root",db="test")
    query='SELECT FrameID FROM videoFrame ORDER BY FrameID DESC LIMIT 1;'
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

def get_videoID():
    conn = mdb.connect(host="localhost",user="root",passwd="root",db="test")
    query='SELECT videoID FROM video_local ORDER BY videoID DESC LIMIT 1;'
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

def getframe(logfilepath):
    cmd = ["grep" ,"Parsed_showinfo_1", logfilepath]
    # with open(os.path.join(videoname,"timestamps"),"wb") as out:
    proc1 = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd2 = ["grep", "n:[0-9]*","-o"]
    proc2 = subprocess.Popen(cmd2,stdin=proc1.stdout,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # cmd3 = ["grep", "[0-9]* \| [0-9]*\.[0-9]*", "-o"]
    # proc3 = subprocess.Popen(cmd3,stdin=proc2.stdout,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_str=proc2.communicate()[0]
    frame=frame_str.replace('n:','').strip().split('\n')
    return frame

def insert_video_local_sql(videoID=0,videoName="none",path="path_path",if_ship="0",if_xh = "0",if_faceagg="0",iffacereg="0",now_time="2017",ifall="0"):
    conn = mdb.connect(host="localhost",user="root",passwd="root",db="test")
    cursor = conn.cursor()
    cursor.execute("select * from video_local")

    cursor.execute("SELECT VERSION()")
    sqlvideo = ("insert into video_local values({:d},\"{:s}\",\"{:s}\",\"{:s}\",\"{:s}\",\"{:s}\",\"{:s}\",\"{:s}\",\"{:s}\")").\
        format(int(videoID),videoName,path,str(if_ship),str(if_xh),str(if_faceagg),str(iffacereg),str(now_time),str(ifall))
    print "(video_local) sql command is ",sqlvideo
    try:
        cursor.execute(sqlvideo)
        conn.commit()
        print "(video_local) write sql success"
    except mdb.Error, e:
        conn.rollback()
        print "(video_local) MySQL Error %d: %s" % (e.args[0], e.args[1])
        # all_flog = -1
    conn.close()
def imgshipdetection(net,img_path,image_name,result_dir,FrameID_now):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image

    im = cv2.imread(img_path)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    rectresults = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        thresh = CONF_THRESH
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        # print "start ploting",image_name
        if len(inds) == 0:
            continue

        img = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(img, aspect='equal')
        class_name = cls
        shipid_now = get_vid()-1
        #############
        FrameID_now = get_FrameID()-1
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            temp = list(dets[i,:])
            temp.append(class_name)
            rectresults.append(temp)
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
            ISOTIMEFORMAT='%Y-%m-%d %X'
            time_det = time.strftime(ISOTIMEFORMAT,time.localtime(time.time() ) )
            print "***time is ",str(time_det)
            print "***image_name, result_dir",image_name,"\n" ,result_dir
            temp_dir = os.path.dirname(img_path)
            print "*****temp_dir",temp_dir
            print "*****img_path",img_path
            insertsql((shipid_now+1),int(FrameID_now +1),int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])\
            ,class_name,score,str(temp_dir),str(time_det))
    plt.axis('off')
    plt.tight_layout()
    if len(rectresults) != 0:
        imageresultpath =  os.path.join(result_dir,"res_"+  image_name + '.png')
        plt.savefig(imageresultpath)
            # plotresult = plotresults(im, cls, dets,image_name, result_dir,CONF_THRESH)
            # if plotresult != -1:
            #     results.extend(plotresult[0])
            #     imageresultpath.append(plotresult[1])
        return rectresults, imageresultpath
    else:
        return -1
# input: net, im_dir, timestamps, videopath
def imglistshipdetection(net, im_dir,timestamps,videopath,this_frameID):
    image_list = os.listdir(im_dir)
    # only png and jpg
    for ficher in image_list[:]:
        if not(ficher.endswith('.png')):
            image_list.remove(ficher)
    video_results = []
    for im_path in image_list:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',im_path
        print 'Demo for data/demo/{}'.format(im_path)
        this_frameID += 1
            # rect, img_save_ptah_name = shipdetect(net, img_path, img_name, save_path)
            # demo(net, imgdir_name, imgdir)
        # img_name = os.path.basename(im_path)
        name, ext = os.path.splitext(im_path)
        resultdir = os.path.join(os.path.dirname(im_dir),os.path.basename(im_dir)+'result')
        im_path = os.path.join(im_dir, im_path)
        if not os.path.exists(resultdir):
            os.mkdir(resultdir)
        print im_path
        imgdetectresult = imgshipdetection(net, im_path,name, resultdir,this_frameID)
        if imgdetectresult != -1:
            print imgdetectresult
            temp = [videopath, imgdetectresult[0],imgdetectresult[1],timestamps[int(name)-1]]
            video_results.extend(temp)
    if len(video_results) != 0:
        return video_results
    else:
        return -1

def keyFrameExtraction(videopath):
    videoname, ext = os.path.splitext(videopath)
    # print videoname, videopath
    keyframesdir = videoname+'scene'
    if not os.path.exists(keyframesdir):
        os.mkdir(keyframesdir)
    cmds = ["/usr/bin/ffmpeg" ,"-i",videopath, "-vf", "select=gt(scene\,0.5),showinfo", "-vsync", "vfr",os.path.join(keyframesdir,"%d.png")]
    # cmds = ["/usr/bin/ffmpeg" ,"-i",videopath, "-vf", "select='eq(pict_type,I)'", "-vsync", "vfr",os.path.join(videoname,"%d.png")]
    # cmds = ["/usr/bin/ffmpeg" ,"-i",videopath, "-vf", "select='eq(pict_type,I)'", "-vsync", "vfr",os.path.join(videoname,"%d.png")]
    logfilepath = os.path.join(keyframesdir,"log.log")
    with open(logfilepath,"wb") as out:
        ffmpeg_p = subprocess.Popen(cmds, stdin=out,stdout=out, stderr=out)
        output = ffmpeg_p.communicate()
        # out.write(output)
    return keyframesdir, logfilepath

#input: logfilepath,string e.g. "/home/fenglei/Downloads/videoscene/log.log" 
#output: timestamps for keyframes, list e.g ['10.644', '40.7407', '52.1187', '73.1731', '82.7827', '121.355', '308.008', '440.54']
def videoInfoExtraction(logfilepath):
    cmd = ["grep" ,"showinfo", logfilepath]
# with open(os.path.join(videoname,"timestamps"),"wb") as out:
    proc1 = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd2 = ["grep", "pts_time:[0-9.]*","-o"]
    proc2 = subprocess.Popen(cmd2,stdin=proc1.stdout,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd3 = ["grep", "[0-9]*\.[0-9]*", "-o"]
    proc3 = subprocess.Popen(cmd3,stdin=proc2.stdout,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timestamps_str=proc3.communicate()[0]
    timestamps=timestamps_str.strip().split('\n')
    # print type(timestamps)
    # print len(timestamps)
    # print timestamps
    # out.write(timestamps_str)
    return timestamps
def write_video_local(videopath):
    videoIDnow = get_videoID()-1
    print "videoIDnow: ",videoIDnow
    video_name = os.path.basename(videopath)
    video_name = video_name[:-4]
    print "video_name: ",video_name
    ISOTIMEFORMAT='%Y-%m-%d %X'
    time_det = time.strftime(ISOTIMEFORMAT,time.localtime(time.time() ) )
    insert_video_local_sql((int(videoIDnow)+1),video_name,videopath,"0","0","0","0",str(time_det),"0")

def write_videoFrame_sql(FrameID,videoID,vPosition,path,if_all = "0"):
    conn = mdb.connect(host="localhost",user="root",passwd="root",db="test")
    cursor = conn.cursor()
    cursor.execute("select * from videoFrame")

    cursor.execute("SELECT VERSION()")
    sqlvideoFrame = ("insert into videoFrame values({:d},{:d},\"{:s}\",\"{:s}\",\"{:s}\")").\
        format(int(FrameID),videoID,vPosition,path,if_all)
    print "(videoFrame) sql command is ",sqlvideoFrame
    try:
        cursor.execute(sqlvideoFrame)
        conn.commit()
        print "(videoFrame) write sql success"
    except mdb.Error, e:
        conn.rollback()
        print "(videoFrame) MySQL Error %d: %s" % (e.args[0], e.args[1])
        # all_flog = -1

def write_videoFrame(videopath,logfilepath):
    FrameID = get_FrameID()
    videoID = get_videoID()-1
    vPosition = getframe(logfilepath)
    print "**vPosition----: ",vPosition
    path = "/home/new/Desktop/project/mkl_shipdetection/py-faster-rcnn/data/frame_img"
    write_videoFrame_sql(FrameID,videoID,vPosition,path,"0")

    return FrameID


def video_detection(net, videopath):
    write_video_local(videopath)

    keyframesdir, logfilepath = keyFrameExtraction(videopath)

    this_frameID = write_videoFrame(videopath,logfilepath)

    timestamps = videoInfoExtraction(logfilepath)
    results = imglistshipdetection(net, keyframesdir, timestamps,videopath,this_frameID)
    return results


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
# if __name__ == '__main__':
def main_video(videopath):
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
    results = video_detection(net, videopath)
    print results,"\nend success"
    return 1
# video_path = "/home/new/Desktop/project/mkl_shipdetection/py-faster-rcnn/data/1.MP4"
# main_video(video_path)