# YOLOv5 π by Ultralytics, GPL-3.0 license
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
from pygame import mixer #mp3μ¬μμ νμν λΌμ΄λΈλ¬λ¦¬ pip install pygame 
mixer.init() #μ΄κΈ°ν

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

def soundplay(filename,length):#mp3wotod ν¨μ  filename-μμ νμΌ μ΄λ¦(νμ₯μX), length=μμ κΈΈμ΄
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
                            print('μ')
                            hp = xyxy #handpoint
                            print(hp[0], 480-hp[1], hp[2], 480-hp[3])  
                        
                        if 'hp' in locals() and 'cup' in label:
                            
                            #soundplay('μ»΅',0.5)
                            #soundplay('μ΄',0.5)
                            #soundplay('μμ κΈ°μ€μΌλ‘',2)
                            
                            cp = xyxy #cuppoint
                            cx = (cp[0] + cp[2]) / 2
                            cy = (480 - cp[1] + 480 - cp[3]) / 2
                            print(cx, cy)
                            
                            if hp[0] > cx and 480 - hp[1] < cy: 
                                print('μ»΅μ΄ μμ κΈ°μ€μΌλ‘ μΌμͺ½μ λκ°μ λ°©ν₯μ μμ΅λλ€.') #3μ¬λΆλ©΄
                                #soundplay('μΌμͺ½μ',3)
                            elif hp[0] < cx and hp[2] > cx and 480 - hp[1] < cy: 
                                print('μ»΅μ΄ μμ κΈ°μ€μΌλ‘ μμͺ½ λ°©ν₯μ μμ΅λλ€.') #2μ¬λΆλ©΄
                                #soundplay('μμͺ½',3)
                            elif hp[2] < cx and 480 - hp[1] < cy: 
                                print('μ»΅μ΄ μμ κΈ°μ€μΌλ‘ μ€λ₯Έμͺ½μ λκ°μ λ°©ν₯μ μμ΅λλ€.') #1μ¬λΆλ©΄
                                #soundplay('μ€λ₯Έμͺ½μ',3)
                            elif hp[0] > cx and 480 - hp[3] < cy and 480 - hp[1] > cy: 
                                print('μ»΅μ΄ μμ κΈ°μ€μΌλ‘ μΌμͺ½ λ°©ν₯μ μμ΅λλ€.') #6μ¬λΆλ©΄
                                #soundplay('μΌμͺ½',3)
                            elif hp[2] < cx and 480 - hp[1] > cy and 480 - hp[3] < cy: 
                                print('μ»΅μ΄ μμ κΈ°μ€μΌλ‘ μ€λ₯Έμͺ½ λ°©ν₯μ μμ΅λλ€.') #4μ¬λΆλ©΄ 
                                #soundplay('μ€λ₯Έμͺ½',3)
                            elif hp[0] > cx and 480 - hp[3] > cy: 
                                print('μ»΅μ΄ μμ κΈ°μ€μΌλ‘ μΌμͺ½μλ λκ°μ λ°©ν₯μ μμ΅λλ€.') #9μ¬λΆλ©΄
                                #soundplay('μΌμͺ½μλ',3)
                            elif hp[0] < cx and hp[2] > cx and 480 - hp[3] > cy: 
                                print('μ»΅μ΄ μμ κΈ°μ€μΌλ‘ λ·μͺ½ λ°©ν₯μ μμ΅λλ€.') #8μ¬λΆλ©΄
                                #soundplay('λ·μͺ½',3)
                            elif hp[2] < cx and 480 - hp[3] > cy: 
                                print('μ»΅μ΄ μμ κΈ°μ€μΌλ‘ μ€λ₯Έμͺ½μλ λκ°μ λ°©ν₯μ μμ΅λλ€.') #7μ¬λΆλ©΄
                                #soundplay('μ€λ₯Έμͺ½μλ',3)
                            del hp
                        
                        if 'hp' in locals() and 'phone' in label:
                             
                            
                            #soundplay('νΈλν°',0.5)
                            #soundplay('μ΄',0.5)
                            #soundplay('μμ κΈ°μ€μΌλ‘',2)
                            
                            pp = xyxy #phonepoint
                            px = (pp[0] + pp[2]) / 2
                            py = (480 - pp[1] + 480 - pp[3]) / 2
                            print(px, py)  
                            
                            if hp[0] > px and 480 - hp[1] < py: 
                                print('νΈλν°μ΄ μμ κΈ°μ€μΌλ‘ μΌμͺ½μ λκ°μ λ°©ν₯μ μμ΅λλ€.') #3μ¬λΆλ©΄
                                #soundplay('μΌμͺ½μ',3)
                            elif hp[0] < px and hp[2] > px and 480 - hp[1] < py: 
                                print('νΈλν°μ΄ μμ κΈ°μ€μΌλ‘ μμͺ½ λ°©ν₯μ μμ΅λλ€.') #2μ¬λΆλ©΄
                                #soundplay('μμͺ½',3)
                            elif hp[2] < px and 480 - hp[1] < py: 
                                print('νΈλν°μ΄ μμ κΈ°μ€μΌλ‘ μ€λ₯Έμͺ½μ λκ°μ λ°©ν₯μ μμ΅λλ€.') #1μ¬λΆλ©΄
                                #soundplay('μ€λ₯Έμͺ½μ',3)
                            elif hp[0] > px and 480 - hp[3] < py and 480 - hp[1] > py: 
                                print('νΈλν°μ΄ μμ κΈ°μ€μΌλ‘ μΌμͺ½ λ°©ν₯μ μμ΅λλ€.') #6μ¬λΆλ©΄
                                #soundplay('μΌμͺ½',3)
                            elif hp[2] < px and 480 - hp[1] > py and 480 - hp[3] < py: 
                                print('νΈλν°μ΄ μμ κΈ°μ€μΌλ‘ μ€λ₯Έμͺ½ λ°©ν₯μ μμ΅λλ€.') #4μ¬λΆλ©΄ 
                                #soundplay('μ€λ₯Έμͺ½',3)
                            elif hp[0] > px and 480 - hp[3] > py: 
                                print('νΈλν°μ΄ μμ κΈ°μ€μΌλ‘ μΌμͺ½μλ λκ°μ λ°©ν₯μ μμ΅λλ€.') #9μ¬λΆλ©΄
                                #soundplay('μΌμͺ½μλ',3)
                            elif hp[0] < px and hp[2] > px and 480 - hp[3] > py: 
                                print('νΈλν°μ΄ μμ κΈ°μ€μΌλ‘ λ·μͺ½ λ°©ν₯μ μμ΅λλ€.') #8μ¬λΆλ©΄
                                #soundplay('λ·μͺ½',3)
                            elif hp[2] < px and 480 - hp[3] > py: 
                                print('νΈλν°μ΄ μμ κΈ°μ€μΌλ‘ μ€λ₯Έμͺ½μλ λκ°μ λ°©ν₯μ μμ΅λλ€.') #7μ¬λΆλ©΄
                                #soundplay('μ€λ₯Έμͺ½μλ',3)
                            del hp

                        else:    
                            if 'cup' in label:
                                
                                #soundplay('μ»΅',0.5)
                                #soundplay('μ΄',0.5)
                                cp = xyxy #cuppoint
                                cx = (cp[0] + cp[2]) / 2
                                cy = (480 - cp[1] + 480 - cp[3]) / 2
                                
                                distancei = (((2*3.14*180)/((cp[2]-cp[0])+(((480-cp[1])-(480-cp[3]))*360)))*1000)+3 #κ°μ²΄κ±°λ¦¬μΈ‘μ 
                                redistancei = distancei*2.54 #μΈμΉ -> μΌμΉλ³ν
                                objectleng = (((redistancei + 100 )*(redistancei - 100))**(1/2))
                                cpix = (cp[2]-cp[0])/16 #λ°μ΄λ©λ°μ€λ₯Ό 16λ±λΆν΄μ£Όκ³  10cmμ§λ¦¬ μ»΅λ 16λ±λΆ(0.625cm)ν΄μ€¬μ΅λλ€.
                                if 320 > cx and 320 > cp[2]:
                                    cx1 = ((320 - cp[2])/cpix)*0.625 #κ°μ΄λ°λ₯Ό κΈ°μ€μΌλ‘ κ°μ²΄λ§νΌ λ¨μ΄μ§ κ±°λ¦¬λ₯Ό cpixλ‘ λλλ€μ 0.625λ₯Ό κ³±νλ©΄ κ±°λ¦¬μ μκ΄μμ΄ κ°μ΄λ°λ₯Ό κΈ°μ€μΌλ‘ μΌλ§λ λ¨μ΄μ Έ μλμ§ μ μ μμ΅λλ€.
                                elif 320 < cx and 320 < cp[0]:
                                    cx1 = ((cp[0] - 320)/cpix)*0.625
                                else:
                                    cx1 = 0
                                if 212 < cx and 426 > cx and 319 < cy: 
                                    print('μ»΅μ΄ μμͺ½ λ°©ν₯μ μμ΅λλ€.') #2μ¬λΆλ©΄
                                    #soundplay('μμͺ½',3)
                                elif 212 < cx and 426 > cx and 160 > cy: 
                                    print('μ»΅μ΄ λ·μͺ½ λ°©ν₯μ μμ΅λλ€.') #8μ¬λΆλ©΄
                                    #soundplay('λ·μͺ½',3)
                                elif 212 > cx:
                                    if 160 > cy: 
                                        print('μ»΅μ΄ μΌμͺ½μλ λκ°μ λ°©ν₯μ μμ΅λλ€.') #9μ¬λΆλ©΄
                                        #soundplay('μΌμͺ½μλ',3)
                                    elif 159 < cy and 320 > cy: 
                                        print('μ»΅μ΄ μΌμͺ½ λ°©ν₯μ μμ΅λλ€.') #6μ¬λΆλ©΄
                                        #soundplay('μΌμͺ½',3)
                                    elif 319 < cy: 
                                        print('μ»΅μ΄ μΌμͺ½μ λκ°μ λ°©ν₯μ μμ΅λλ€.') #3μ¬λΆλ©΄
                                        #soundplay('μΌμͺ½μ',3)
                                elif 425 < cx:
                                    if 160 > cy:
                                        print('μ»΅μ΄ μ€λ₯Έμͺ½μλ λκ°μ λ°©ν₯μ μμ΅λλ€.') #7μ¬λΆλ©΄
                                        #soundplay('μ€λ₯Έμͺ½μλ',3)
                                    elif 159 < cy and 320 > cy: 
                                        print('μ»΅μ΄ μ€λ₯Έμͺ½ λ°©ν₯μ μμ΅λλ€.') #4μ¬λΆλ©΄ 
                                        #soundplay('μ€λ₯Έμͺ½',3)
                                    elif 319 < cy: 
                                        print('μ»΅μ΄ μ€λ₯Έμͺ½μ λκ°μ λ°©ν₯μ μμ΅λλ€.') #1μ¬λΆλ©΄
                                        #soundplay('μ€λ₯Έμͺ½μ',3)
                                else :
                                    print('μ»΅μ΄ κ°μ΄λ°λ°©ν₯μ μμ΅λλ€.')
                                    #soundplay('κ°μ΄λ°',3)


                                if redistancei < 30:
                                    print('μ½ 30cm λ΄μ μμ΅λλ€.')
                                    #soundplay('30cmλ΄',3)
                                elif redistancei > 29 and redistancei < 40:
                                    print('μ½ 40cm κ±°λ¦¬μ μμ΅λλ€.')
                                    #soundplay('40cm',3)
                                elif redistancei > 39 and redistancei < 50:
                                    print('μ½ 50cm κ±°λ¦¬μ μμ΅λλ€.')
                                    #soundplay('50cm',3)
                                elif redistancei > 49 and redistancei < 60:
                                    print('μ½ 60cm κ±°λ¦¬μ μμ΅λλ€.')
                                    #soundplay('60cm',3)
                                elif redistancei > 59 and redistancei < 70:
                                    print('μ½ 70cm κ±°λ¦¬μ μμ΅λλ€.')
                                    #soundplay('70cm',3)
                                elif redistancei > 69 and redistancei < 80:
                                    print('μ½ 80cm κ±°λ¦¬μ μμ΅λλ€.')
                                    #soundplay('80cm',3)
                                elif redistancei > 79 and redistancei < 90:
                                    print('μ½ 90cm κ±°λ¦¬μ μμ΅λλ€.')
                                    #soundplay('90cm',3)
                                elif redistancei > 89 and redistancei < 100:
                                    print('μ½ 1m κ±°λ¦¬μ μμ΅λλ€.')
                                    #soundplay('1m',3)
                                elif redistancei > 99:
                                    print('μ½ 1m μΈμ μμ΅λλ€.')
                                    #soundplay('1mμΈ',3)
                                print(cx, cy)
                                print(objectleng, redistancei)
                                print('{} cm'.format(cx1))

                            if 'phone' in label:
                                 
                                #soundplay('νΈλν°',0.5)
                                #soundplay('μ΄', 0.5)
                                pp = xyxy #phonepoint
                                px = (pp[0] + pp[2]) / 2
                                py = (480 - pp[1] + 480 - pp[3]) / 2
                                  
                                distancei = (((2*3.14*180)/((pp[2]-pp[0])+(((480-pp[1])-(480-pp[3]))*360)))*1000)+3
                                redistancei = distancei*2.54 #μΉ΄λ©λΌλΆν° κ°μ²΄κΉμ§μ κ±°λ¦¬
                                objectleng = (((redistancei + 100 )*(redistancei - 100))**(1/2)) #λ°λ₯ κ±°λ¦¬(2.5mλ₯Ό κΈ°μ€μΌλ‘ νλ©΄ μΊ νμ§μ΄ λ?μ 1mλ₯Ό κΈ°μ€μΌλ‘ μ‘μμ΅λλ€.)
                                ppix = (pp[2]-pp[0])/16 #λ°μ΄λ©λ°μ€λ₯Ό 16λ±λΆν΄μ£Όκ³  8cmμ§λ¦¬ νΈλν°λ 16λ±λΆ(0.5cm)ν΄μ€¬μ΅λλ€.
                                if 320 > px and 320 > pp[2]:
                                    px1 = ((320 - pp[2])/ppix)*0.5 #κ°μ΄λ°λ₯Ό κΈ°μ€μΌλ‘ κ°μ²΄λ§νΌ λ¨μ΄μ§ κ±°λ¦¬λ₯Ό ppixλ‘ λλλ€μ 0.5λ₯Ό κ³±νλ©΄ κ±°λ¦¬μ μκ΄μμ΄ κ°μ΄λ°λ₯Ό κΈ°μ€μΌλ‘ μΌλ§λ λ¨μ΄μ Έ μλμ§ μ μ μμ΅λλ€.
                                elif 320 < px and 320 < pp[0]:
                                    px1 = ((pp[0] - 320)/ppix)*0.5
                                else:
                                    px1 = 0
                                
                                if 212 < px and 426 > px and 319 < py: 
                                    print('νΈλν°μ΄ μμͺ½ λ°©ν₯μ μμ΅λλ€.') #2μ¬λΆλ©΄
                                    #soundplay('μμͺ½',3)
                                elif 212 < px and 426 > px and 160 > py: 
                                    print('νΈλν°μ΄ λ·μͺ½ λ°©ν₯μ μμ΅λλ€.') #8μ¬λΆλ©΄
                                    #soundplay('λ·μͺ½',3)
                                elif 212 > px:
                                    if 160 > py: 
                                        print('νΈλν°μ΄ μΌμͺ½μλ λκ°μ λ°©ν₯μ μμ΅λλ€.') #9μ¬λΆλ©΄
                                        #soundplay('μΌμͺ½μλ',3)
                                    elif 159 < py and 320 > py: 
                                        print('νΈλν°μ΄ μΌμͺ½ λ°©ν₯μ μμ΅λλ€.') #6μ¬λΆλ©΄
                                        #soundplay('μΌμͺ½',3)
                                    elif 319 < py: 
                                        print('νΈλν°μ΄ μΌμͺ½μ λκ°μ λ°©ν₯μ μμ΅λλ€.') #3μ¬λΆλ©΄
                                        #soundplay('μΌμͺ½μ',3)
                                elif 425 < px:
                                    if 160 > py:
                                        print('νΈλν°μ΄ μ€λ₯Έμͺ½μλ λκ°μ λ°©ν₯μ μμ΅λλ€.') #7μ¬λΆλ©΄
                                        #soundplay('μ€λ₯Έμͺ½μλ',3)
                                    elif 159 < py and 320 > py: 
                                        print('νΈλν°μ΄ μ€λ₯Έμͺ½ λ°©ν₯μ μμ΅λλ€.') #4μ¬λΆλ©΄ 
                                        #soundplay('μ€λ₯Έμͺ½',3)
                                    elif 319 < py: 
                                        print('νΈλν°μ΄ μ€λ₯Έμͺ½μ λκ°μ λ°©ν₯μ μμ΅λλ€.') #1μ¬λΆλ©΄
                                        #soundplay('μ€λ₯Έμͺ½μ',3)
                                else :
                                    print('νΈλν°μ΄ κ°μ΄λ°λ°©ν₯μ μμ΅λλ€.')
                                    #soundplay('κ°μ΄λ°',3)

                                if redistancei < 30:
                                    print('μ½ 30cm λ΄μ μμ΅λλ€.')
                                    #soundplay('30cmλ΄',3)
                                elif redistancei > 29 and redistancei < 40:
                                    print('μ½ 40cm κ±°λ¦¬μ μμ΅λλ€.')
                                    #soundplay('40cm',3)
                                elif redistancei > 39 and redistancei < 50:
                                    print('μ½ 50cm κ±°λ¦¬μ μμ΅λλ€.')
                                    #soundplay('50cm',3)
                                elif redistancei > 49 and redistancei < 60:
                                    print('μ½ 60cm κ±°λ¦¬μ μμ΅λλ€.')
                                    #soundplay('60cm',3)
                                elif redistancei > 59 and redistancei < 70:
                                    print('μ½ 70cm κ±°λ¦¬μ μμ΅λλ€.')
                                    #soundplay('70cm',3)
                                elif redistancei > 69 and redistancei < 80:
                                    print('μ½ 80cm κ±°λ¦¬μ μμ΅λλ€.')
                                    #soundplay('80cm',3)
                                elif redistancei > 79 and redistancei < 90:
                                    print('μ½ 90cm κ±°λ¦¬μ μμ΅λλ€.')
                                    #soundplay('90cm',3)
                                elif redistancei > 89 and redistancei < 100:
                                    print('μ½ 1m κ±°λ¦¬μ μμ΅λλ€.')
                                    #soundplay('1m',3)
                                elif redistancei > 99:
                                    print('μ½ 1m μΈμ μμ΅λλ€.')
                                    #soundplay('1mμΈ',3)
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