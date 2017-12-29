from ship_detect import main_detect
from demo_video import main_video

def main_all_ship(input_list):
    FrameID = 1

    jpg_list = []
    mp4_list = []
    for one_element in input_list:
        jpg_list.append(one_element)
        if one_element[-1] == '4':
            main_video(one_element)
        elif one_element[-1] == 'g':
            main_detect(one_element,FrameID)
        else :
            print "wrong input (not .mp4 or .jpg)"



input_list = ['/home/new/Desktop/project/mkl_shipdetection/py-faster-rcnn/data/1.MP4','/home/new/Desktop/project/mkl_shipdetection/py-faster-rcnn/data/copy2.MP4']
main_all_ship(input_list)
#'/home/new/Desktop/project/mkl_shipdetection/py-faster-rcnn/data/1.MP4'
# '/home/new/Desktop/project/mkl_shipdetection/py-faster-rcnn/data/demo/free2.jpg',