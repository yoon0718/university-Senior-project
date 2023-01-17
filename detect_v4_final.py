# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
from asyncio.windows_events import NULL
#from operator import truediv
import os
import sys
from pathlib import Path


import cv2
import torch
import torch.backends.cudnn as cudnn

import time
from pygame import mixer #mp3재생에 필요한 라이브러리 pip install pygame 
mixer.init() #초기화

#from playsound import playsound

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

def soundplay(filename,length):#mp3wotod 함수  filename-음원 파일 이름(확장자X), length=음원 길이
    mixer.music.load('./sound/'+filename+'.mp3')
    mixer.music.play()
    time.sleep(length)
    mixer.music.stop()

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False, # show results 
        #save_txt=False,  # save results to *.txt
        save_txt=False,
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):   
                    if save_txt:  # Write to file
                        xywh= (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        if save_crop:
                           save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    
                        
                        if 'hand' in label:
                            print('손')
                            hp = xyxy #handpoint
                            print(hp[0], 480-hp[1], hp[2], 480-hp[3])  
                        
                        if 'hp' in locals() and 'cup' in label:
                            
                            #soundplay('컵',0.5)
                            #soundplay('이',0.5)
                            #soundplay('손을 기준으로',2)
                            
                            cp = xyxy #cuppoint
                            cx = (cp[0] + cp[2]) / 2
                            cy = (480 - cp[1] + 480 - cp[3]) / 2
                            print(cx, cy)
                            
                            if hp[0] > cx and 480 - hp[1] < cy: 
                                print('컵이 손을 기준으로 왼쪽위 대각선방향에 있습니다.') #3사분면
                                #soundplay('왼쪽위',3)
                            elif hp[0] < cx and hp[2] > cx and 480 - hp[1] < cy: 
                                print('컵이 손을 기준으로 앞쪽 방향에 있습니다.') #2사분면
                                #soundplay('앞쪽',3)
                            elif hp[2] < cx and 480 - hp[1] < cy: 
                                print('컵이 손을 기준으로 오른쪽위 대각선방향에 있습니다.') #1사분면
                                #soundplay('오른쪽위',3)
                            elif hp[0] > cx and 480 - hp[3] < cy and 480 - hp[1] > cy: 
                                print('컵이 손을 기준으로 왼쪽 방향에 있습니다.') #6사분면
                                #soundplay('왼쪽',3)
                            elif hp[2] < cx and 480 - hp[1] > cy and 480 - hp[3] < cy: 
                                print('컵이 손을 기준으로 오른쪽 방향에 있습니다.') #4사분면 
                                #soundplay('오른쪽',3)
                            elif hp[0] > cx and 480 - hp[3] > cy: 
                                print('컵이 손을 기준으로 왼쪽아래 대각선방향에 있습니다.') #9사분면
                                #soundplay('왼쪽아래',3)
                            elif hp[0] < cx and hp[2] > cx and 480 - hp[3] > cy: 
                                print('컵이 손을 기준으로 뒷쪽 방향에 있습니다.') #8사분면
                                #soundplay('뒷쪽',3)
                            elif hp[2] < cx and 480 - hp[3] > cy: 
                                print('컵이 손을 기준으로 오른쪽아래 대각선방향에 있습니다.') #7사분면
                                #soundplay('오른쪽아래',3)
                            del hp
                        
                        if 'hp' in locals() and 'phone' in label:
                             
                            
                            #soundplay('핸드폰',0.5)
                            #soundplay('이',0.5)
                            #soundplay('손을 기준으로',2)
                            
                            pp = xyxy #phonepoint
                            px = (pp[0] + pp[2]) / 2
                            py = (480 - pp[1] + 480 - pp[3]) / 2
                            print(px, py)  
                            
                            if hp[0] > px and 480 - hp[1] < py: 
                                print('핸드폰이 손을 기준으로 왼쪽위 대각선방향에 있습니다.') #3사분면
                                #soundplay('왼쪽위',3)
                            elif hp[0] < px and hp[2] > px and 480 - hp[1] < py: 
                                print('핸드폰이 손을 기준으로 앞쪽 방향에 있습니다.') #2사분면
                                #soundplay('앞쪽',3)
                            elif hp[2] < px and 480 - hp[1] < py: 
                                print('핸드폰이 손을 기준으로 오른쪽위 대각선방향에 있습니다.') #1사분면
                                #soundplay('오른쪽위',3)
                            elif hp[0] > px and 480 - hp[3] < py and 480 - hp[1] > py: 
                                print('핸드폰이 손을 기준으로 왼쪽 방향에 있습니다.') #6사분면
                                #soundplay('왼쪽',3)
                            elif hp[2] < px and 480 - hp[1] > py and 480 - hp[3] < py: 
                                print('핸드폰이 손을 기준으로 오른쪽 방향에 있습니다.') #4사분면 
                                #soundplay('오른쪽',3)
                            elif hp[0] > px and 480 - hp[3] > py: 
                                print('핸드폰이 손을 기준으로 왼쪽아래 대각선방향에 있습니다.') #9사분면
                                #soundplay('왼쪽아래',3)
                            elif hp[0] < px and hp[2] > px and 480 - hp[3] > py: 
                                print('핸드폰이 손을 기준으로 뒷쪽 방향에 있습니다.') #8사분면
                                #soundplay('뒷쪽',3)
                            elif hp[2] < px and 480 - hp[3] > py: 
                                print('핸드폰이 손을 기준으로 오른쪽아래 대각선방향에 있습니다.') #7사분면
                                #soundplay('오른쪽아래',3)
                            del hp

                        else:    
                            if 'cup' in label:
                                
                                #soundplay('컵',0.5)
                                #soundplay('이',0.5)
                                cp = xyxy #cuppoint
                                cx = (cp[0] + cp[2]) / 2
                                cy = (480 - cp[1] + 480 - cp[3]) / 2
                                
                                distancei = (((2*3.14*180)/((cp[2]-cp[0])+(((480-cp[1])-(480-cp[3]))*360)))*1000)+3 #객체거리측정
                                redistancei = distancei*2.54 #인치 -> 센치변환
                                objectleng = (((redistancei + 100 )*(redistancei - 100))**(1/2))
                                cpix = (cp[2]-cp[0])/16 #바운딩박스를 16등분해주고 10cm짜리 컵도 16등분(0.625cm)해줬습니다.
                                if 320 > cx and 320 > cp[2]:
                                    cx1 = ((320 - cp[2])/cpix)*0.625 #가운데를 기준으로 객체만큼 떨어진 거리를 cpix로 나눈다음 0.625를 곱하면 거리에 상관없이 가운데를 기준으로 얼마나 떨어져 있는지 알 수 있습니다.
                                elif 320 < cx and 320 < cp[0]:
                                    cx1 = ((cp[0] - 320)/cpix)*0.625
                                else:
                                    cx1 = 0
                                if 212 < cx and 426 > cx and 319 < cy: 
                                    print('컵이 앞쪽 방향에 있습니다.') #2사분면
                                    #soundplay('앞쪽',3)
                                elif 212 < cx and 426 > cx and 160 > cy: 
                                    print('컵이 뒷쪽 방향에 있습니다.') #8사분면
                                    #soundplay('뒷쪽',3)
                                elif 212 > cx:
                                    if 160 > cy: 
                                        print('컵이 왼쪽아래 대각선방향에 있습니다.') #9사분면
                                        #soundplay('왼쪽아래',3)
                                    elif 159 < cy and 320 > cy: 
                                        print('컵이 왼쪽 방향에 있습니다.') #6사분면
                                        #soundplay('왼쪽',3)
                                    elif 319 < cy: 
                                        print('컵이 왼쪽위 대각선방향에 있습니다.') #3사분면
                                        #soundplay('왼쪽위',3)
                                elif 425 < cx:
                                    if 160 > cy:
                                        print('컵이 오른쪽아래 대각선방향에 있습니다.') #7사분면
                                        #soundplay('오른쪽아래',3)
                                    elif 159 < cy and 320 > cy: 
                                        print('컵이 오른쪽 방향에 있습니다.') #4사분면 
                                        #soundplay('오른쪽',3)
                                    elif 319 < cy: 
                                        print('컵이 오른쪽위 대각선방향에 있습니다.') #1사분면
                                        #soundplay('오른쪽위',3)
                                else :
                                    print('컵이 가운데방향에 있습니다.')
                                    #soundplay('가운데',3)


                                if redistancei < 30:
                                    print('약 30cm 내에 있습니다.')
                                    #soundplay('30cm내',3)
                                elif redistancei > 29 and redistancei < 40:
                                    print('약 40cm 거리에 있습니다.')
                                    #soundplay('40cm',3)
                                elif redistancei > 39 and redistancei < 50:
                                    print('약 50cm 거리에 있습니다.')
                                    #soundplay('50cm',3)
                                elif redistancei > 49 and redistancei < 60:
                                    print('약 60cm 거리에 있습니다.')
                                    #soundplay('60cm',3)
                                elif redistancei > 59 and redistancei < 70:
                                    print('약 70cm 거리에 있습니다.')
                                    #soundplay('70cm',3)
                                elif redistancei > 69 and redistancei < 80:
                                    print('약 80cm 거리에 있습니다.')
                                    #soundplay('80cm',3)
                                elif redistancei > 79 and redistancei < 90:
                                    print('약 90cm 거리에 있습니다.')
                                    #soundplay('90cm',3)
                                elif redistancei > 89 and redistancei < 100:
                                    print('약 1m 거리에 있습니다.')
                                    #soundplay('1m',3)
                                elif redistancei > 99:
                                    print('약 1m 외에 있습니다.')
                                    #soundplay('1m외',3)
                                print(cx, cy)
                                print(objectleng, redistancei)
                                print('{} cm'.format(cx1))

                            if 'phone' in label:
                                 
                                #soundplay('핸드폰',0.5)
                                #soundplay('이', 0.5)
                                pp = xyxy #phonepoint
                                px = (pp[0] + pp[2]) / 2
                                py = (480 - pp[1] + 480 - pp[3]) / 2
                                  
                                distancei = (((2*3.14*180)/((pp[2]-pp[0])+(((480-pp[1])-(480-pp[3]))*360)))*1000)+3
                                redistancei = distancei*2.54 #카메라부터 객체까지의 거리
                                objectleng = (((redistancei + 100 )*(redistancei - 100))**(1/2)) #바닥 거리(2.5m를 기준으로 하면 캠화질이 낮아 1m를 기준으로 잡았습니다.)
                                ppix = (pp[2]-pp[0])/16 #바운딩박스를 16등분해주고 8cm짜리 핸드폰도 16등분(0.5cm)해줬습니다.
                                if 320 > px and 320 > pp[2]:
                                    px1 = ((320 - pp[2])/ppix)*0.5 #가운데를 기준으로 객체만큼 떨어진 거리를 ppix로 나눈다음 0.5를 곱하면 거리에 상관없이 가운데를 기준으로 얼마나 떨어져 있는지 알 수 있습니다.
                                elif 320 < px and 320 < pp[0]:
                                    px1 = ((pp[0] - 320)/ppix)*0.5
                                else:
                                    px1 = 0
                                
                                if 212 < px and 426 > px and 319 < py: 
                                    print('핸드폰이 앞쪽 방향에 있습니다.') #2사분면
                                    #soundplay('앞쪽',3)
                                elif 212 < px and 426 > px and 160 > py: 
                                    print('핸드폰이 뒷쪽 방향에 있습니다.') #8사분면
                                    #soundplay('뒷쪽',3)
                                elif 212 > px:
                                    if 160 > py: 
                                        print('핸드폰이 왼쪽아래 대각선방향에 있습니다.') #9사분면
                                        #soundplay('왼쪽아래',3)
                                    elif 159 < py and 320 > py: 
                                        print('핸드폰이 왼쪽 방향에 있습니다.') #6사분면
                                        #soundplay('왼쪽',3)
                                    elif 319 < py: 
                                        print('핸드폰이 왼쪽위 대각선방향에 있습니다.') #3사분면
                                        #soundplay('왼쪽위',3)
                                elif 425 < px:
                                    if 160 > py:
                                        print('핸드폰이 오른쪽아래 대각선방향에 있습니다.') #7사분면
                                        #soundplay('오른쪽아래',3)
                                    elif 159 < py and 320 > py: 
                                        print('핸드폰이 오른쪽 방향에 있습니다.') #4사분면 
                                        #soundplay('오른쪽',3)
                                    elif 319 < py: 
                                        print('핸드폰이 오른쪽위 대각선방향에 있습니다.') #1사분면
                                        #soundplay('오른쪽위',3)
                                else :
                                    print('핸드폰이 가운데방향에 있습니다.')
                                    #soundplay('가운데',3)

                                if redistancei < 30:
                                    print('약 30cm 내에 있습니다.')
                                    #soundplay('30cm내',3)
                                elif redistancei > 29 and redistancei < 40:
                                    print('약 40cm 거리에 있습니다.')
                                    #soundplay('40cm',3)
                                elif redistancei > 39 and redistancei < 50:
                                    print('약 50cm 거리에 있습니다.')
                                    #soundplay('50cm',3)
                                elif redistancei > 49 and redistancei < 60:
                                    print('약 60cm 거리에 있습니다.')
                                    #soundplay('60cm',3)
                                elif redistancei > 59 and redistancei < 70:
                                    print('약 70cm 거리에 있습니다.')
                                    #soundplay('70cm',3)
                                elif redistancei > 69 and redistancei < 80:
                                    print('약 80cm 거리에 있습니다.')
                                    #soundplay('80cm',3)
                                elif redistancei > 79 and redistancei < 90:
                                    print('약 90cm 거리에 있습니다.')
                                    #soundplay('90cm',3)
                                elif redistancei > 89 and redistancei < 100:
                                    print('약 1m 거리에 있습니다.')
                                    #soundplay('1m',3)
                                elif redistancei > 99:
                                    print('약 1m 외에 있습니다.')
                                    #soundplay('1m외',3)
                                print(px, py)
                                print(objectleng, redistancei)
                                print('{} cm'.format(px1))

                            

                                

                        
            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)