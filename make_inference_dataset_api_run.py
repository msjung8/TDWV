#!/usr/bin/env python
# coding: utf-8

# # 순서
# ### import modules
# ### methods
# ### predictor 초기화
# ### 전처리

# In[1]:


import os


# In[2]:


get_ipython().system('pwd')


# In[3]:


get_ipython().system('ls ./datasets/new_datasets/')


# In[4]:


get_ipython().system('rm -rf ./datasets/new_datasets/ ')


# In[5]:


get_ipython().system('ipython make_inference_dataset_decompo.py')


# # import modules

# In[6]:


# inference
import torch
import numpy as np
import cv2
import detectron2
import imutils
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import json

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import Instances
from detectron2.data.transforms import Resize
from detectron2.engine import DefaultPredictor

#pose estimation
from PIL import Image
import PIL 
import tqdm
setup_logger()
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo


# 

# # methods 선언
# 
# cal_margin, set_margin, cal_by_x, onlykeeppersonclass

# In[7]:


def cal_margin(a1, a2, m, i, im):
    if a1 - m <= 0 :  #왼쪽 부족
        rest_a = m-a1 #이게 지금 문제, x<=m일때, rest_x = m-(m-x)
        a1 = 0
        #print("1", rest_a)
        #오른쪽이 충분하건 부족하건 끝내면 됨
        if a2+ rest_a+ m >= im.shape[i]: #오른쪽 부족 - 왼쪽은 이미 부족
            w = im.shape[i]-a1
            a2 = im.shape[i]
            rest_a = a2+rest_a+m - im.shape[i]

        else: #오른쪽 충분
            a2 = a2+rest_a+m
            rest_a = 0



    elif a1 - m > 0: #왼쪽 충분
        a1 = a1-m
        rest_a = 0 #rest_x는 0

        #오른쪽이 부족할경우 다시 수행해야함.
        if a2+ rest_a+ m >= im.shape[i]: #오른쪽 부족
            rest_a = m- (im.shape[i] - a2) #오른쪽 처리못하고 남은 마진
            a2 = im.shape[i]

            if a1-rest_a <=0: #오른쪽에서 남은 마진을 왼쪽에서 처리하기에 부족 - 이제 양쪽 모두 불가능 - 그대로 끝
                a1=0
                rest_a = rest_a - a1
            else: #왼쪽 안부족 - 왼쪽에서 남은 마진만큼 더 빼서 설정.
                a1=a1-rest_a
                rest_a = 0

        else: #오른쪽 충분
            rest_a = 0
            a2 = a2 + rest_a+m

    tmp = a2-a1

    return a1, a2, tmp
        

def set_margin(x, y, x2, y2, m, im): #cal_margin 사용
    #x point조정 - 양쪽을 늘리는것을 최우선으로 하자. 그게 안될경우 한쪽 한쪽, 둘다 안될경우 이렇게 나누기
    #연산을 할때, 우려되는 경우의수 왼쪽부족->오른쪽부족 , 왼쪽충분->오른쪽부족 마지막에 rest_x가 남게됨
    #이를 어떻게 해결할까 - 왼쪽부족->오른쪽부족은 그냥 그대로 냅두면 됨
    #왼충 오부는 왼쪽에 다시 할당을 하러 오면 된다.
    #print("before margin x,y,x2,y2 = ", x, y, x2, y2)
    #w길이에 대해서 다시 봐야한다.
    x,x2,w = cal_margin(x,x2,m,1,im)
    y,y2,h = cal_margin(y,y2,m,0,im)
    
    #print("x,y,x2,y2, w, h", x, y, x2, y2, w, h)
    
    return x,y,x2,y2,w,h


#사용하는 메소드
def cal_by_x(x,x2,w,i,im): # i는 im.shape[i]의 비교대상 - height는 고정, width 기준 비교라면 i=0
    margin_left = (192-w)/2 # 중앙점에서 한쪽의 w/2 를 뺴고 남은 앞으로 더 빼야할 양 -> 이값이 음수면 
    margin_right = (192-w)/2 # 중앙점에서 한쪽의 w/2 뺴고 남은 앞으로 더 뺴야할 양
    

    if x - margin_left < 0:  # x-margin_left가 이미지 왼쪽 벽보다 더 가면 x = 0으로 놔야함 , margin_left도 남는다. 
        margin_transfer = margin_left - x
        margin_left = 0
        margin_right += margin_transfer
        x = 0

        if x2 + margin_right > im.shape[i]: # x2 + margin_right가 이미지 오른쪽 벽보다 더 가면 x2 = im.shape[1]로 놔야함. margin_right도 남음 but 처리불가
            # 0 ~ im.shape[1]
            margin_right = x2 + margin_right - im.shape[i]
            x2 = im.shape[i]

            #margin처리를 더 할수 없는 상황이다. 이럴때는 어떻게 처리해야할까?? 일단 그냥처리 // 다른방법은.. 비율로만 맞추는방법 그후


        else: # x2 + margin_right 그대로 x2에 저장, margin은 더이상 남지 않는다.
            # 0 ~ 192 (x2 + margin_right)
            x2 = x2 + margin_right
            margin_right=0
            margin_transfer=0

    else: # x > margin_left일 경우, 둘의 차가 이미지 왼쪽 벽에 닿지 않음 즉 margin_left는 남지 않는다
        x = x - margin_left
        margin_left=0

        if x2 + margin_right > im.shape[i]: # 이미지 오른쪽 벽을 넘어감
            # x-margin_left ~ im.shape[1]
            margin_transfer = x2 + margin_right - im.shape[i]
            margin_right = 0
            x2 = im.shape[i]

            if x - margin_transfer < 0: #왼쪽 벽 막힘
                margin_transfer=0
                x=0
            else:
                x = x - margin_transfer
                margin_transfer=0

        else: # 이미지 오른쪽 안넘음
            # x-margin_left ~ x2 + margin_right
            x2 = x2 + margin_right
            margin_right = 0
    #print("x2 - x = w", x2, x, w)
    w = x2 - x

    return x,x2,w


def onlykeeppersonclass(outputs, im, edge_im,label_im): #set_margin -> cal_margin
    
    # 해당하는 마스크만 남김.
    #print("instances", outputs['instances'].pred_classes[0])
    
    num_masks = outputs['instances'].pred_masks.shape[0]
    #print(outputs['instances'])
    min_false_elements = np.inf
    min_false_idx = None
    
    for i in range(num_masks):
        mask = outputs['instances'].pred_masks[i]
        num_false_elements = np.sum(mask == 'False')
        if num_false_elements < min_false_elements:
            min_false_elements = num_false_elements
            min_false_idx = i
    
    #masks = outputs['instances'].pred_masks[min_false_idx:min_false_idx+1]
    
    
    #print(outputs['instances'].pred_masks.shape)
    
    
    #class가 사람인것 + bbox 넓이 가장 큰것 선택
    max_index=0
    max_area=0
    #print("outputs",outputs['instances'])
    for i,bbox in enumerate(outputs['instances'].pred_boxes): #x,y x2, y2
        #선택한 class가 사람인가?
        if int(outputs['instances'].pred_classes[i]) != 0: 
            continue
            
        #선택한 bbox가 넓이가 가장 큰가?
        if max_area == 0:
            max_index = i
            max_area = (bbox[3]-bbox[1])*(bbox[2]-bbox[0]) # x2 - x1 * y2 - y1
            #print("max:",max_area)
            continue
        area = (bbox[3]-bbox[1])*(bbox[2]-bbox[0])
        #print("area",area)
        if max_area < area:
            max_index = i
            max_area = area
    
    
    ##왼쪽이 불충분한 mask##
    masks = outputs['instances'].pred_masks[max_index:max_index+1]
    cls = outputs['instances'].pred_classes[max_index]
    scores = outputs["instances"].scores[max_index]
    boxes = outputs['instances'].pred_boxes[max_index]
    
    
    #print(boxes)
    for i, box in enumerate(boxes):
    # get the x, y, width, and height of the bbox
        x, y, x2, y2 = box.tolist()
    
    x,y,x2,y2 = int(x), int(y), int(x2), int(y2)
    ##print("box x, y, x2, y2",x,y,x2,y2)
    
    
    #바로 resize하기전에 앞서,이 이미지의 id에 해당하는 annotation을 불러오고
    
    
    
    #불러온 annotation의 bbox를 아래에 나올 ratio에 맞게 같이 resize 해야한다.
    
    #헌데 지금 생각나는 이미지 불러오는방식은 파일의 이미지-> 동일한 image_id의 annotation을 참고해 seg를 crop인데,
    #지금 index 들어가는 방식이 [0] 후에 id를 하나만 접근 가능해 이 annotation을 먼저 참고해서 image_id가 같은 아이템을 변경해야함
    
    
    
    
    
    
    #1. 먼저 margin을 준다. - set_margin
    m = 10 #margin
    x,y,x2,y2,w,h = set_margin(x,y,x2,y2,m,im)
    
 
    ##print("after margin x,y,x2,y2 = ", x, y, x2, y2)
    ##print("after margin x, y, w, h ", x, y, w, h)
    
    #2. resize -> cal_by_x, cal_by_y를 통해 width x height를 192 x 256으로 고정한다.
    
    ratio = 256 / h 
    
    ##print("before resize img size =", im.shape)
    im = imutils.resize(im, height=int(im.shape[0]*ratio))
    edge_im = imutils.resize(edge_im, height=int(edge_im.shape[0]*ratio))
    label_im = imutils.resize(label_im, height=int(label_im.shape[0]*ratio))
    ##print("after resize img size =", im.shape)
    
    w = round(w*ratio)
    h = round(h*ratio)
    ##print("after resize bbox w = ", w)
    ##print("after resize bbox h = ", h)
    
    x = int(x*ratio)
    y = int(y*ratio)
    x2 = int(x2*ratio)
    y2 = int(y2*ratio)
    
    ##print("after resize bbox x, y", x,y)
    
    
        
    # im width가 192를 넘는지 확인해야 한다.
    if im.shape[1] < 192: #앞으로 height를 256에 맞게 조정 - 조정이라기보단 그냥 원점에서 잘라낸다. (무조건 256보다 큰 길이로 resize되서)
        print("resized image width is less than 192")
        ratio = 192 / im.shape[1]
        im = imutils.resize(im, width=int(im.shape[1]*ratio))
        edge_im = imutils.resize(edge_im, width=int(edge_im.shape[1]*ratio))
        label_im = imutils.resize(label_im, width=int(label_im.shape[1]*ratio))
        w = round(im.shape[1]*ratio) #예의상 쓴것 192임
        h = 256

        x = 0
        y = int(y*ratio)
        x2 = 192
        y2 = int(y+256)
      ##  print("after resize bbox x, y", x,y)
        w = x2-x
        
    else: # 앞으로 width를 192에 맞게 조정
        x,x2,w = cal_by_x(x,x2,w,1,im)
        
    
    masks = masks.unsqueeze(0)
    masks = F.interpolate(masks.float(), size=(im.shape[0], im.shape[1]), mode="nearest").bool()
    masks = masks.squeeze(0)
    
    
    
    #위의 과정에 annotation의 변환 또한 적용해야 한다.
    
    
    # slicing image and mask
#     im = im[y:h, x:w]
#     masks = masks[:, y:h, x:w]
    
#     print(im.shape)
#     im = cv2.resize(im1, dsize=(192, 256), interpolation=cv2.INTER_AREA)
    
    
    # 마스크 리사이징
    
    #     masks = resize_transform(masks)
    
    # remove all other classes which are not person(index:0)
    indx_to_remove = (cls != 0).nonzero().flatten().tolist()
    
    
    # delete corresponding arrays
    cls = np.delete(cls.cpu().numpy(), indx_to_remove)
    scores = np.delete(scores.cpu().numpy(), indx_to_remove)
    masks = np.delete(masks.cpu().numpy(), indx_to_remove, axis=0)
#     boxes = np.delete(boxes.cpu()numpy(), indx_to_remove, axis=0)

    # convert back to tensor and move to cuda
    cls = torch.tensor(cls).to('cuda:0')
    scores = torch.tensor(scores).to('cuda:0')
    masks = torch.tensor(masks).to('cuda:0')
#     boxes = torch.tensor(boxes).to('cuda:0')
    
    # not interested in boxes
    outputs['instances'].remove('pred_boxes')
    outputs['instances'].remove('scores')
    outputs['instances'].remove('pred_classes')
    outputs['instances'].remove('pred_masks')

    # create new instance obj and set its fields
    obj = detectron2.structures.Instances(image_size=(im.shape[0], im.shape[1]))
    obj.set('pred_classes', cls)
    obj.set('scores', scores)
    obj.set('pred_masks', masks)
    
    
    x,y,w,h = int(x), int(y), int(w), int(h)
   ## print("final: x y w h", x, y, w, h)
    return obj, x, y, w, h, im, edge_im, label_im


# In[8]:


def onlykeeppersonclass_api_1(outputs, im, edge_im,label_im): #set_margin -> cal_margin
    
    # 해당하는 마스크만 남김.
    #print("instances", outputs['instances'].pred_classes[0])
    
    num_masks = outputs['instances'].pred_masks.shape[0]
    #print(outputs['instances'])
    min_false_elements = np.inf
    min_false_idx = None
    
    for i in range(num_masks):
        mask = outputs['instances'].pred_masks[i]
        num_false_elements = np.sum(mask == 'False')
        if num_false_elements < min_false_elements:
            min_false_elements = num_false_elements
            min_false_idx = i
    
    #masks = outputs['instances'].pred_masks[min_false_idx:min_false_idx+1]
    
    
    #print(outputs['instances'].pred_masks.shape)
    
    
    #class가 사람인것 + bbox 넓이 가장 큰것 선택
    max_index=0
    max_area=0
    #print("outputs",outputs['instances'])
    for i,bbox in enumerate(outputs['instances'].pred_boxes): #x,y x2, y2
        #선택한 class가 사람인가?
        if int(outputs['instances'].pred_classes[i]) != 0: 
            continue
            
        #선택한 bbox가 넓이가 가장 큰가?
        if max_area == 0:
            max_index = i
            max_area = (bbox[3]-bbox[1])*(bbox[2]-bbox[0]) # x2 - x1 * y2 - y1
            #print("max:",max_area)
            continue
        area = (bbox[3]-bbox[1])*(bbox[2]-bbox[0])
        #print("area",area)
        if max_area < area:
            max_index = i
            max_area = area
    
    
    ##왼쪽이 불충분한 mask##
    masks = outputs['instances'].pred_masks[max_index:max_index+1]
    cls = outputs['instances'].pred_classes[max_index]
    scores = outputs["instances"].scores[max_index]
    boxes = outputs['instances'].pred_boxes[max_index]
    
    
    #print(boxes)
    for i, box in enumerate(boxes):
    # get the x, y, width, and height of the bbox
        x, y, x2, y2 = box.tolist()
    
    x,y,x2,y2 = int(x), int(y), int(x2), int(y2)
    ##print("box x, y, x2, y2",x,y,x2,y2)
    
    
    #바로 resize하기전에 앞서,이 이미지의 id에 해당하는 annotation을 불러오고
    
    
    
    #불러온 annotation의 bbox를 아래에 나올 ratio에 맞게 같이 resize 해야한다.
    
    #헌데 지금 생각나는 이미지 불러오는방식은 파일의 이미지-> 동일한 image_id의 annotation을 참고해 seg를 crop인데,
    #지금 index 들어가는 방식이 [0] 후에 id를 하나만 접근 가능해 이 annotation을 먼저 참고해서 image_id가 같은 아이템을 변경해야함
    
    
    
    
    
    
    #1. 먼저 margin을 준다. - set_margin
    m = 10 #margin
    x,y,x2,y2,w,h = set_margin(x,y,x2,y2,m,im)
    
 
    ##print("after margin x,y,x2,y2 = ", x, y, x2, y2)
    ##print("after margin x, y, w, h ", x, y, w, h)
    
    #2. resize -> cal_by_x, cal_by_y를 통해 width x height를 192 x 256으로 고정한다.
    
    ratio = 256 / h 
    
    ##print("before resize img size =", im.shape)
    im = imutils.resize(im, height=int(im.shape[0]*ratio))
    edge_im = imutils.resize(edge_im, height=int(edge_im.shape[0]*ratio))
    label_im = imutils.resize(label_im, height=int(label_im.shape[0]*ratio))
    ##print("after resize img size =", im.shape)
    
    w = round(w*ratio)
    h = round(h*ratio)
    ##print("after resize bbox w = ", w)
    ##print("after resize bbox h = ", h)
    
    x = int(x*ratio)
    y = int(y*ratio)
    x2 = int(x2*ratio)
    y2 = int(y2*ratio)
    
    ##print("after resize bbox x, y", x,y)
    
    
        
    # im width가 192를 넘는지 확인해야 한다.
    if im.shape[1] < 192: #앞으로 height를 256에 맞게 조정 - 조정이라기보단 그냥 원점에서 잘라낸다. (무조건 256보다 큰 길이로 resize되서)
        print("resized image width is less than 192")
        ratio = 192 / im.shape[1]
        im = imutils.resize(im, width=int(im.shape[1]*ratio))
        edge_im = imutils.resize(edge_im, width=int(edge_im.shape[1]*ratio))
        label_im = imutils.resize(label_im, width=int(label_im.shape[1]*ratio))
        w = round(im.shape[1]*ratio) #예의상 쓴것 192임
        h = 256

        x = 0
        y = int(y*ratio)
        x2 = 192
        y2 = int(y+256)
      ##  print("after resize bbox x, y", x,y)
        w = x2-x
        
    else: # 앞으로 width를 192에 맞게 조정
        x,x2,w = cal_by_x(x,x2,w,1,im)
        
    
    masks = masks.unsqueeze(0)
    masks = F.interpolate(masks.float(), size=(im.shape[0], im.shape[1]), mode="nearest").bool()
    masks = masks.squeeze(0)
    
    
    
    #위의 과정에 annotation의 변환 또한 적용해야 한다.
    
    
    # slicing image and mask
#     im = im[y:h, x:w]
#     masks = masks[:, y:h, x:w]
    
#     print(im.shape)
#     im = cv2.resize(im1, dsize=(192, 256), interpolation=cv2.INTER_AREA)
    
    
    # 마스크 리사이징
    
    #     masks = resize_transform(masks)
    
    # remove all other classes which are not person(index:0)
    indx_to_remove = (cls != 0).nonzero().flatten().tolist()
    
    
    # delete corresponding arrays
    cls = np.delete(cls.cpu().numpy(), indx_to_remove)
    scores = np.delete(scores.cpu().numpy(), indx_to_remove)
    masks = np.delete(masks.cpu().numpy(), indx_to_remove, axis=0)
#     boxes = np.delete(boxes.cpu()numpy(), indx_to_remove, axis=0)

    # convert back to tensor and move to cuda
    cls = torch.tensor(cls).to('cuda:0')
    scores = torch.tensor(scores).to('cuda:0')
    masks = torch.tensor(masks).to('cuda:0')
#     boxes = torch.tensor(boxes).to('cuda:0')
    
    # not interested in boxes
    #outputs['instances'].remove('pred_boxes')
    #outputs['instances'].remove('scores')
    #outputs['instances'].remove('pred_classes')
    #outputs['instances'].remove('pred_masks')

    # create new instance obj and set its fields
    obj = detectron2.structures.Instances(image_size=(im.shape[0], im.shape[1]))
    obj.set('pred_classes', cls)
    obj.set('scores', scores)
    obj.set('pred_masks', masks)
    
    
    x,y,w,h = int(x), int(y), int(w), int(h)
   ## print("final: x y w h", x, y, w, h)
    return obj, x, y, w, h, im, edge_im, label_im


# In[9]:


def onlykeeppersonclass_onlyimg(outputs, im): #set_margin -> cal_margin
    
    # 해당하는 마스크만 남김.
    #print("instances", outputs['instances'].pred_classes[0])
    
    num_masks = outputs['instances'].pred_masks.shape[0]
    #print(outputs['instances'])
    min_false_elements = np.inf
    min_false_idx = None
    
    for i in range(num_masks):
        mask = outputs['instances'].pred_masks[i]
        num_false_elements = np.sum(mask == 'False')
        if num_false_elements < min_false_elements:
            min_false_elements = num_false_elements
            min_false_idx = i
    
    #masks = outputs['instances'].pred_masks[min_false_idx:min_false_idx+1]
    
    
    #print(outputs['instances'].pred_masks.shape)
    
    
    #class가 사람인것 + bbox 넓이 가장 큰것 선택
    max_index=0
    max_area=0
    #print("outputs",outputs['instances'])
    for i,bbox in enumerate(outputs['instances'].pred_boxes): #x,y x2, y2
        #선택한 class가 사람인가?
        if int(outputs['instances'].pred_classes[i]) != 0: 
            continue
            
        #선택한 bbox가 넓이가 가장 큰가?
        if max_area == 0:
            max_index = i
            max_area = (bbox[3]-bbox[1])*(bbox[2]-bbox[0]) # x2 - x1 * y2 - y1
            #print("max:",max_area)
            continue
        area = (bbox[3]-bbox[1])*(bbox[2]-bbox[0])
        #print("area",area)
        if max_area < area:
            max_index = i
            max_area = area
    
    
    ##왼쪽이 불충분한 mask##
    masks = outputs['instances'].pred_masks[max_index:max_index+1]
    cls = outputs['instances'].pred_classes[max_index]
    scores = outputs["instances"].scores[max_index]
    boxes = outputs['instances'].pred_boxes[max_index]
    
    
    #print(boxes)
    for i, box in enumerate(boxes):
    # get the x, y, width, and height of the bbox
        x, y, x2, y2 = box.tolist()
    
    x,y,x2,y2 = int(x), int(y), int(x2), int(y2)
    ##print("box x, y, x2, y2",x,y,x2,y2)
    
    
    #바로 resize하기전에 앞서,이 이미지의 id에 해당하는 annotation을 불러오고
    
    
    
    #불러온 annotation의 bbox를 아래에 나올 ratio에 맞게 같이 resize 해야한다.
    
    #헌데 지금 생각나는 이미지 불러오는방식은 파일의 이미지-> 동일한 image_id의 annotation을 참고해 seg를 crop인데,
    #지금 index 들어가는 방식이 [0] 후에 id를 하나만 접근 가능해 이 annotation을 먼저 참고해서 image_id가 같은 아이템을 변경해야함
    
    
    
    
    
    
    #1. 먼저 margin을 준다. - set_margin
    m = 10 #margin
    x,y,x2,y2,w,h = set_margin(x,y,x2,y2,m,im)
    
 
    ##print("after margin x,y,x2,y2 = ", x, y, x2, y2)
    ##print("after margin x, y, w, h ", x, y, w, h)
    
    #2. resize -> cal_by_x, cal_by_y를 통해 width x height를 192 x 256으로 고정한다.
    
    ratio = 256 / h 
    
    ##print("before resize img size =", im.shape)
    im = imutils.resize(im, height=int(im.shape[0]*ratio))
    ##print("after resize img size =", im.shape)
    
    w = round(w*ratio)
    h = round(h*ratio)
    ##print("after resize bbox w = ", w)
    ##print("after resize bbox h = ", h)
    
    x = int(x*ratio)
    y = int(y*ratio)
    x2 = int(x2*ratio)
    y2 = int(y2*ratio)
    
    ##print("after resize bbox x, y", x,y)
    
    
        
    # im width가 192를 넘는지 확인해야 한다.
    if im.shape[1] < 192: #앞으로 height를 256에 맞게 조정 - 조정이라기보단 그냥 원점에서 잘라낸다. (무조건 256보다 큰 길이로 resize되서)
        print("resized image width is less than 192")
        ratio = 192 / im.shape[1]
        im = imutils.resize(im, width=int(im.shape[1]*ratio))
        w = round(im.shape[1]*ratio) #예의상 쓴것 192임
        h = 256

        x = 0
        y = int(y*ratio)
        x2 = 192
        y2 = int(y+256)
      ##  print("after resize bbox x, y", x,y)
        w = x2-x
        
    else: # 앞으로 width를 192에 맞게 조정
        x,x2,w = cal_by_x(x,x2,w,1,im)
        
    
    masks = masks.unsqueeze(0)
    masks = F.interpolate(masks.float(), size=(im.shape[0], im.shape[1]), mode="nearest").bool()
    masks = masks.squeeze(0)
    
    
    
    #위의 과정에 annotation의 변환 또한 적용해야 한다.
    
    
    # slicing image and mask
#     im = im[y:h, x:w]
#     masks = masks[:, y:h, x:w]
    
#     print(im.shape)
#     im = cv2.resize(im1, dsize=(192, 256), interpolation=cv2.INTER_AREA)
    
    
    # 마스크 리사이징
    
    #     masks = resize_transform(masks)
    
    # remove all other classes which are not person(index:0)
    indx_to_remove = (cls != 0).nonzero().flatten().tolist()
    
    
    # delete corresponding arrays
    cls = np.delete(cls.cpu().numpy(), indx_to_remove)
    scores = np.delete(scores.cpu().numpy(), indx_to_remove)
    masks = np.delete(masks.cpu().numpy(), indx_to_remove, axis=0)
#     boxes = np.delete(boxes.cpu()numpy(), indx_to_remove, axis=0)

    # convert back to tensor and move to cuda
    cls = torch.tensor(cls).to('cuda:0')
    scores = torch.tensor(scores).to('cuda:0')
    masks = torch.tensor(masks).to('cuda:0')
#     boxes = torch.tensor(boxes).to('cuda:0')
    
    # not interested in boxes
    outputs['instances'].remove('pred_boxes')
    outputs['instances'].remove('scores')
    outputs['instances'].remove('pred_classes')
    outputs['instances'].remove('pred_masks')

    # create new instance obj and set its fields
    obj = detectron2.structures.Instances(image_size=(im.shape[0], im.shape[1]))
    obj.set('pred_classes', cls)
    obj.set('scores', scores)
    obj.set('pred_masks', masks)
    
    
    x,y,w,h = int(x), int(y), int(w), int(h)
   ## print("final: x y w h", x, y, w, h)
    return obj, x, y, w, h, im


# #  predictor 초기화

# In[10]:


#main
# Set up the configuration file
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the threshold for detections

# Set up the predictor
predictor = DefaultPredictor(cfg)


# # pose estimation

# In[11]:


class Detector:

    def __init__(self, model_type = 'keypointsDetection'):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu" # cpu or cuda

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, image):
        predictions = self.predictor(image)
        
        #오류 위치
        #print(predictions)
        if len(predictions['instances']) == 0:
            return 0
        max_box = predictions['instances'][0].pred_boxes.tensor.cpu().numpy()[0]
        max_index=0
        max_area=0
        for i in range(0, len(predictions['instances'])):
            box = predictions['instances'][0].pred_boxes.tensor.cpu().numpy()[0]
            box_area = (box[2]-box[0])*(box[3]-box[1])

            #첫 번째에는 바로 max영역에 넓이 넣는다. 이후부터 비교
            if i==0:
                max_area = box_area
                continue

            if box_area > max_area:
                max_area = box_area
                max_box = box
                max_index=i
        
        viz = Visualizer(image[:,:,::-1],metadata= MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),scale=1)#instance_mode=ColorMode.IMAGE_BW)        
        output = viz.draw_instance_predictions(predictions['instances'][max_index].to('cpu'))
        #print(predictions['instances'][max_index])
        filename = 'result.jpg'
        #cv2.imwrite(filename, output.get_image()[:,:,::-1])
        plt.imshow(output.get_image()[:,:,::-1])
        plt.show()
        pose_img = output.get_image()[:,:,::-1]
        return pose_img
detector = Detector()


# In[12]:



def get_pose_info_dt2(img_path_dir):
    image = cv2.imread(img_path_dir)
    predictions = detector.predictor(image)
    
    #print(predictions['instances'][0].pred_boxes.tensor.numpy()[0])
    try:
        max_box = predictions['instances'][0].pred_boxes.tensor.numpy()[0]
        max_index=0
        max_area=0
        for i in range(0, len(predictions['instances'])):
            box = predictions['instances'][0].pred_boxes.tensor.numpy()[0]
            box_area = (box[2]-box[0])*(box[3]-box[1])

            #첫 번째에는 바로 max영역에 넓이 넣는다. 이후부터 비교
            if i==0:
                max_area = box_area
                continue

            if box_area > max_area:
                max_area = box_area
                max_box = box
                max_index=i
        
        #print(max_index)
        #print(max_box)
        
        #이 좌표를 이제 json으로 사용하면 된다.
        s_index = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]        
        s_pred = []
        #print(predictions['instances'].pred_keypoints.numpy()[0][s_index])
        for i in s_index:
            s_pred.append(predictions['instances'].pred_keypoints.numpy()[max_index][i])
        #오른어깨 6번, 왼어깨 5번
        tmp = (s_pred[6]+s_pred[5])/2
        #print("tmp :", tmp)
        s_pred.insert(1, tmp)

        
        #print(torch.from_numpy(np.array(s_pred)).flatten().numpy())

        points = torch.from_numpy(np.array(s_pred)).flatten().numpy() #coco 18 key 신버전
        #print(torch.tensor(predictions['instances'].pred_keypoints.numpy()[max_index]).flatten().numpy())
        #points = torch.tensor(predictions['instances'].pred_keypoints.numpy()[max_index]).flatten().numpy() #구버전
        return points
    except:
        print(img_path_dir+" cannot detect instance")
        return 0
    

def make_annot(img_path_dir, pose_save_dir):
    #이미지 파일을 전부 가져와서 리스트로 만들고

    file_list = os.listdir(img_path_dir) #test_img의 파일을 전부 가져온다.
    
    #가져온 이미지들을 각각 get_pose_info한다.
    
    #delete_file(img_path_dir+'/'+'.ipynb_checkpoints') #실행 할 때, 파일 오류 일으키는 주범 삭제 따로 shell에서 해줘야한다..
    for f in sorted(file_list):
        #print(img_path_dir+'/'+f)
        if isinstance(get_pose_info_dt2(img_path_dir+'/'+f) , int):#있다면 배열, 없다면 int다. #keypoint가 검출되지않으면 이미지 삭제 후 continue 진행
            #delete_file(img_path_dir+'/'+f)
            continue
        
        keypoints = get_pose_info_dt2(img_path_dir+'/'+f).tolist()
        content = {"version": 1.0, "people": [{"face_keypoints": [], 
                                               "pose_keypoints": keypoints, 
                                               "hand_right_keypoints": [], "hand_left_keypoints": []}]}

        with open(pose_save_dir+f.replace('.jpg','_keypoints.json'),'w') as json_file:
            json.dump(content,json_file)          
            
        #print(f+' to json complete')

def make_annot2(img_path_dir, img_name,  pose_save_dir):
    #이미지 하나만 씀

    if isinstance(get_pose_info_dt2(img_path_dir+'/'+img_name) , int):#있다면 배열, 없다면 int다. #keypoint가 검출되지않으면 이미지 삭제 후 continue 진행
        #delete_file(img_path_dir+'/'+ img_name)
        return 0

    keypoints = get_pose_info_dt2(img_path_dir+'/'+img_name).tolist()
    content = {"version": 1.0, "people": [{"face_keypoints": [], 
                                           "pose_keypoints": keypoints, 
                                           "hand_right_keypoints": [], "hand_left_keypoints": []}]}

    with open(pose_save_dir+img_name.replace('.jpg','_keypoints.json'),'w') as json_file:
        json.dump(content,json_file)          

        #print(f+' to json complete')
    return 1


# # 전처리 순서
# -----
# ## 1.  얻은 BBox에서 pose estimation 수행하는 코드, human segmentation 코드
# -----
# ## 2. 얻은 BBox에서 배경 subtraction 사용해서 edge 정보 얻기
# -----
# ## 3.  192 256 BBox 얻기 
# ###  bbox 영역 찾기 -> detectron2를 통해 predict bbox 찾기
# ##  margin 부여 -> bbox 기존 크기에서 margin만큼 더허준다 ( 상하좌우 )
# ##  bbox h 256 기준 resize, w 192 기준 조정
# -----
# 
# 
# 

# # 이 이후로 이미지 생성처리가 시작됨

# # 경로 설정
# ## test_img에 이미지 최소 1개 업로드 후 시작

# In[13]:


#이미지 directory path
all_img_dir_path = os.getcwd() + '/datasets/new_datasets'
#가져온 이미지 directory
#print(all_img_dir_path)
img_dir = os.path.join(all_img_dir_path,'test_img/')
img_nobg_dir = os.path.join(all_img_dir_path,'test_img_nobg/')
img_onlybg_dir = os.path.join(all_img_dir_path,'test_img_onlybg/')
img_clothes_dir = os.path.join(all_img_dir_path,'test_clothes/')
img_clothes_upper_dir = os.path.join(all_img_dir_path,'test_clothes_upper_resized/')
img_clothes_lower_dir = os.path.join(all_img_dir_path,'test_clothes_lower_resized/')
img_clothes_upper_edge_dir = os.path.join(all_img_dir_path,'test_clothes_upper_resized_edge/')
img_clothes_lower_edge_dir = os.path.join(all_img_dir_path,'test_clothes_lower_resized_edge/')
img_clothes_upper_color_dir = os.path.join(all_img_dir_path,'test_clothes_upper_resized_color/')
img_clothes_lower_color_dir = os.path.join(all_img_dir_path,'test_clothes_lower_resized_color/')

#이미지 192 256 save path
img_save_dir = os.path.join(all_img_dir_path, 'test_img_resized/')
edge_save_dir = os.path.join(all_img_dir_path, 'test_edge/')
edge_resized_save_dir = os.path.join(all_img_dir_path, 'test_edge_resized/')
#edge_resized_upper_dir = 
#edge_resized_lower_dir

color_save_dir = os.path.join(all_img_dir_path, 'test_color/')
color_resized_save_dir = os.path.join(all_img_dir_path, 'test_color_resized/')
color_bg_resized_save_dir = os.path.join(all_img_dir_path, 'test_color_bg_resized/')
pose_save_dir = os.path.join(all_img_dir_path, 'test_pose/')
pose_resized_save_dir = os.path.join(all_img_dir_path, 'test_pose_resized/')
label_save_dir = os.path.join(all_img_dir_path, 'test_label/')
label_resized_save_dir = os.path.join(all_img_dir_path, 'test_label_resized/')


pose_resized_img_save_dir = os.path.join(all_img_dir_path, 'test_pose_resized_img/') #포즈 이미지 확인용 dir


# In[14]:


# detectron2 dir에서 실행.


# In[15]:


def upload_img(img_dir, img_save_dir, img_name):
    test_img = cv2.imread(img_dir+img_name)
    if img_name.endswith('.png'):
        img_name = img_name.replace('.png', '.jpg')
    if img_name.endswith('.jpeg'):
        img_name = img_name.replace('.jpeg', '.jpg')
    cv2.imwrite(img_save_dir+img_name, test_img)
    return test_img, img_name


# label, edge 생성

# In[16]:


origin_dir = './datasets/raw_data/'
img_name = 'yn_01.png'


# In[17]:


def api_one(origin_dir, test_img_dir, origin_img_name): #이미지 정보를 받고나서 resized된 이미지들 , annot 처리
    #%cd datasets/new_datasets/test_img
    test_img, img_name = upload_img(origin_dir, test_img_dir, origin_img_name)
    #img_name = origin_img_name
    
    
    #edge 만들기
    get_ipython().run_line_magic('cd', 'U_2_Net')
    import u2net_human_seg_TDWV as human_seg
    # 원본 이미지 human segmentation test_edge에 먼저 저장
    bs = human_seg.background_subtractor(img_dir, edge_save_dir)
    bs.main()
    get_ipython().run_line_magic('cd', '../')

    
    
    ### 이미지
    
    edge_img = cv2.imread(edge_save_dir+img_name.replace('.png','.jpg'))
    plt.imshow(edge_img)
    plt.show()
    
    _ , edge_img_th = cv2.threshold(edge_img, 128, 255, cv2.THRESH_BINARY) #128 보다 큰것 255으로 만듬 
    plt.imshow(edge_img_th)
    plt.show()
    
    edge_img_th_not = cv2.bitwise_not(edge_img_th) # 255를 0으로, 127이하는 128이상.
    plt.imshow(edge_img_th_not)
    plt.show()
    
    _, edge_img_th_not_th = cv2.threshold(edge_img_th_not, 128, 255, cv2.THRESH_BINARY)
    plt.imshow(edge_img_th_not_th)
    plt.show()
    
    white_mask = np.zeros_like(test_img)*255
    
    test2_img = np.where(edge_img_th_not_th==(0,0,0), test_img, edge_img_th_not_th) #사람 배경분리 확실해진 edge
    plt.imshow(test2_img)
    plt.show()
    
    #test3_img = np.where(edge_img_th_not_th==(255,255,255), edge_img_th_not_th, test_img)
    #plt.imshow(test3_img)
    #plt.show()
    
    cv2.imwrite(img_nobg_dir+img_name.replace('.png','.jpg'), test2_img)
    #label  원본이미지 배경 제거 후에 수행한다 -> 후에 resize
    #get_ipython().run_line_magic('cd', 'datasets')
    get_ipython().run_line_magic('cd', 'Self-Correction-Human-Parsing-for-ACGPN')
    get_ipython().system('rm -rf ../datasets/new_datasets/test_img/.ipynb_checkpoints')
    get_ipython().system('python3 simple_extractor2.py --input-dir ../datasets/new_datasets/test_img_nobg/ --output-dir ../datasets/new_datasets/test_label/ --model-restore ./checkpoints/exp-schp-201908261155-lip.pth')
    get_ipython().run_line_magic('cd', '../')
    
    label_img = cv2.imread(label_save_dir + img_name.replace('.png','.jpg'))
    #plt.imshow(img)
    #plt.show()
    
    #이미지의 BBox를 먼저 구하고, BBox의 사이즈를 기준으로 192 256으로 resize한다.
    outputs = predictor(test_img)
    if len(outputs['instances'].pred_boxes) == 0:
        print("다시 찍어주세요 cause : no box for this image ->", tmp_name)
        return 0
        
    #print(outputs['instances'][0].pred_boxes.tensor.numpy([0])
    #print(outputs['instances'][0].pred_boxes.tensor.cpu().numpy())
    
    
    #image resize
    obj, x, y, w, h, test_img, _, _ = onlykeeppersonclass_api_1(outputs, test_img, edge_img, label_img)
    obj, x, y, w, h, test2_img, edge_img, label_img = onlykeeppersonclass_api_1(outputs, test2_img, edge_img, label_img)
    
    #사람만있는것과 배경만 있는것으로 분리해야한다.
    test_img_person_resized = test2_img[y:y+h, x:x+w] ##가져가야할 그림
    
    
    test_img_bg_resized = np.where(test2_img == (255,255,255), test_img, (255,255,255))[y:y+h, x:x+w] ##가져가야할 그림
    
    label_img_resized = label_img[y:y+h,x:x+w] ## 가져가야할 그림
    edge_img_resized = edge_img[y:y+h, x:x+w]
    edge_img_resized_not = cv2.bitwise_not(edge_img_resized)
    
    plt.imshow(test_img_person_resized)
    plt.show()
    plt.imshow(test_img_bg_resized)
    plt.show()
    
    #resized img, pose, label
    img_name = origin_img_name.replace('.png','.jpg')
    cv2.imwrite(img_save_dir+img_name, test_img_person_resized)
    cv2.imwrite(img_onlybg_dir+img_name, test_img_bg_resized)
    cv2.imwrite(label_resized_save_dir+img_name, label_img_resized)
    #pose 만들기
    make_annot2(img_save_dir,img_name, pose_resized_save_dir)
    
    return test_img_person_resized, test_img_bg_resized, label_img_resized, pose_resized_save_dir+img_name
api_one(origin_dir, img_dir, img_name)


# tensor로 읽어서 변형함 

# In[19]:


def api_two(origin_dir, test_clothes_dir, origin_img_name): #옷 이미지 , 옷 seg 이미지, 이미지 이름
    #%cd datasets/new_datasets/test_img
    test_img,img_name = upload_img(origin_dir, test_clothes_dir, origin_img_name) #jpg로 찍힘
    
    #test_img = cv2.imread(test_img_dir+test_img_name)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
    get_ipython().system('rm -rf ./datasets/new_datasets/test_img/.ipynb_checkpoints')
    get_ipython().system('ipython clothes_seg.py')
    
    #img_name = origin_img_name.replace('.png','.jpg')
    
    #옷 이미지
    test_clothes = cv2.imread(img_clothes_dir+img_name.replace('.jpg','_generated.png')) 
    test_clothes = cv2.cvtColor(test_clothes, cv2.COLOR_BGR2RGB)
    plt.imshow(test_clothes)
    plt.show()
    
    #먼저 threshold부터 잡자. - 그전에 색부터 잡기
    #필살기
    mask = np.zeros_like(test_clothes)
    mask_one = np.ones_like(test_clothes)*255
    
    
    test_clothes_gray = cv2.cvtColor(test_clothes, cv2.COLOR_BGR2GRAY)
    _, test_clothes_th = cv2.threshold(test_clothes_gray, 1, 255, cv2.THRESH_BINARY) # 옷이 흰색이 될 것임. 배경 검정.
    
    plt.imshow(test_clothes_th)
    plt.show()
    
    
    
    #옷 흰색
    print("threshold 1 255")
    test_clothes_th = cv2.cvtColor(test_clothes_th, cv2.COLOR_GRAY2BGR)
    #plt.imshow(test_clothes_th)
    #plt.show()
    
    #배경 흰색
    print("threshold 1 255 not ") 
    test_clothes_th_not =cv2.bitwise_not(test_clothes_th)
    #plt.imshow(test_clothes_th_not)
    #plt.show()
    
    #옷 전체
    test_clothes_th_all = np.where(test_clothes_th_not == 0, test_img, mask_one)
    print("threshold 1 255 not all")
    #plt.imshow(test_clothes_th_all)
    #plt.show()
    
    
    #---
    #상의 상의는 뭐로해야하지..
    print("dst0")
    dst0 = cv2.inRange(test_clothes, (128,0,0), (255,100,100))
    plt.imshow(dst0)
    plt.show()
    if 255 in dst0:
        print("상의 감지")
        _, dst0 = cv2.threshold(dst0, 1, 255, cv2.THRESH_BINARY)
        #edge
        dst0 = cv2.cvtColor(dst0, cv2.COLOR_GRAY2BGR)
        plt.imshow(dst0)
        plt.show()
        test_clothes_upper_edge = cv2.resize(dst0, (192,256), cv2.INTER_LINEAR)
        
        #color
        dst0 = np.where(dst0==255, test_clothes_th_all, 0)
        plt.imshow(dst0)
        plt.show()
        test_clothes_upper_color = cv2.resize(dst0, (192,256), cv2.INTER_LINEAR)
        
        test_clothes_upper_edge = cv2.cvtColor(test_clothes_upper_edge, cv2.COLOR_RGB2BGR)
        test_clothes_upper_color = cv2.cvtColor(test_clothes_upper_color, cv2.COLOR_RGB2BGR)
        
        #for y in range(len(test_clothes_upper_edge)):
        #    for x in range(len(test_clothes_upper_edge)):
        #        #print(cls_mask_thresh[y][x])
        #        if((test_clothes_upper_edge[y][x] == [255,255,255]).all()):
#       #             print(cls_mask_thresh[y][x])
        #            y_list.append(y)
        #            x_list.append(x)
        #min_bx = max(min(x_list)-10, 0)
        #min_by = max(min(y_list)-10, 0)
        #max_bx2 = min(max(x_list)+10, 192)
        #max_by2 = min(max(y_list)+10, 256)
        
        #edge_cls_crop = test_clothes_upper_edge[min_by:max_by2, min_bx:max_bx2] 
        #plt.imshow(edge_cls_crop)
        #plt.show()
        #edge cls scale
        #print("edge_scaled_cls")
        #test_clothes_upper_edge_resized = cv2.resize(edge_cls_crop,(192,256))
        
        
        #print(min_by, max_by2, min_bx, max_bx2)
        
        cv2.imwrite(img_clothes_upper_edge_dir +img_name , test_clothes_upper_edge)
        cv2.imwrite(img_clothes_upper_color_dir +img_name , test_clothes_upper_color)
        
    
    
    #plt.imshow(test_clothes_upper)
    #plt.show()
    
    #상의 엣지
    
    #상의 컬러
    
    #---
    #하의
    print("dst1")
    dst1 = cv2.inRange(test_clothes, (0, 128, 0), (100, 255, 100))
    if 255 in dst1:
        print("하의 감지")
        red = np.full(test_img.shape, ([255,0,0]), dtype=np.uint8)
        test_clothes_lower= cv2.subtract(test_clothes, red)
        #plt.imshow(test_clothes_lower)
        #plt.show()

        #하의 엣지
        test_clothes_lower_edge = cv2.cvtColor(test_clothes_lower, cv2.COLOR_BGR2GRAY)
        _, test_clothes_lower_edge = cv2.threshold(test_clothes_lower_edge, 1, 255, cv2.THRESH_BINARY)
        test_clothes_lower_edge = cv2.cvtColor(test_clothes_lower_edge, cv2.COLOR_GRAY2BGR)
        plt.imshow(test_clothes_lower_edge)
        plt.show()
        
        #하의 컬러
        test_clothes_lower_color = np.where(test_clothes_lower_edge==255, test_clothes_th_all,0)
        plt.imshow(test_clothes_lower_color)
        plt.show()
        
        #resize save
        test_clothes_lower_edge = cv2.resize(test_clothes_lower_edge, (192,256), cv2.INTER_LINEAR)
        test_clothes_lower_color = cv2.resize(test_clothes_lower_color, (192,256), cv2.INTER_LINEAR)
        
        test_clothes_lower_edge = cv2.cvtColor(test_clothes_lower_edge, cv2.COLOR_RGB2BGR)
        test_clothes_lower_color = cv2.cvtColor(test_clothes_lower_color, cv2.COLOR_RGB2BGR)
        
        
        cv2.imwrite(img_clothes_lower_edge_dir +img_name, test_clothes_lower_edge)
        cv2.imwrite(img_clothes_lower_color_dir +img_name, test_clothes_lower_color)
        
        
    #resized edge upper/lower, color upper/lower
    
api_two(origin_dir,img_clothes_dir, img_name)

