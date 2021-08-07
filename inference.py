import argparse
import importlib
import torch
import utils

import os
import sys

import torchvision , torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from rcnn_transfrom import InterpolationTransform as it

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2

def get_instance_segmentation_model(num_classes,pretrained=True,mode='bilinear'):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    model.transform = it(min_size=(800,), max_size=1333,image_mean=[0.485, 0.456, 0.406],image_std=[0.229, 0.224, 0.225],mode=mode)
    return model

import torchvision
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_ann(img,scores,boxes,masks=None,nms=True,iou_threshold=0.5,conf_threshold=0.4 ,mask_threshold=0.5): # pil , tensor,tensor,float

    dis_rgbs = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128),
        (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128), (255, 255, 255), (0, 0, 0)]
    msk  = masks is not None
    size = img.size[::-1]

    # nms
    if nms :
        idx = torchvision.ops.nms(boxes,scores,iou_threshold=0.3)
        boxes  = boxes[idx]
        scores = scores[idx]
        if msk :
            masks = masks[idx]
    if conf_threshold :
        condition = scores>conf_threshold
        boxes  = boxes[condition]
        scores = scores[condition]   
        if msk :
            masks = masks[condition]
    if msk :
        if  'cuda' in masks.device.type:
            masks = masks.clone().detach().cpu()
        summask = np.zeros(size+(3,),dtype='uint8')
        for i,mask in enumerate(masks): 
            condi      = np.array(mask>0.5)
            condi_3    = np.concatenate([condi.reshape(size+(1,) )]*3,axis=2)
            colorMask  = np.array(Image.new('RGB',(size[1],size[0]),dis_rgbs[i]) )
            colorMask  = colorMask * condi_3 

            # multiply 1/2 if overlapped        
#             overlapped = (colorMask.sum(0) * summask.sum(0)).astype('bool')
#             overlapped = np.stack([overlapped,overlapped,overlapped]).reshape( num_condi.shape+(3,))
#             overMask   = colorMask*overlapped//2 + summask*overlapped//2
#             summask    = summask*(~overlapped) + colorMask*(~overlapped)
#             summask    = summask + overMask
            summask    = summask + (colorMask)
        # summask brightness
#         overlapped = 
#         imgnp , summask = np.array(img)[overlapped]//2 + summask[overlapped]//2
        imgnp           = np.array(img)
        img = Image.fromarray(imgnp+summask ) 
    
    annimg = img.copy()
    draw = ImageDraw.Draw(annimg)
    font = ImageFont.load_default()
    
    for i,box in enumerate(boxes):
        # draw.rectangle(xy=[0,0,150,150],outline=dis_rgb[2] )#, outline=label_color_map[det_labels[i]])
        box_xy = box.tolist()
        draw.rectangle(xy=box_xy,outline=dis_rgbs[i%len(dis_rgbs)] )#, outline=label_color_map[det_labels[i]])
        title = str(round(scores[i].item(),4))
        draw.text(xy=[box_xy[0]+len(title),box_xy[1]-1],text=title,fill='white',font=font)
    del draw
        
    
    return annimg

if __name__ == '__main__' :
    exam_code = '''
    python inference.py -i ./input.jpg -o ./output.jpg
    '''
    desc   = 'Inference'
    parser = argparse.ArgumentParser(f"{desc} Mask R-CNN model",epilog=exam_code)
    # setting
    # -m ~.pth , all(all of models in ./models)
    # -o dir ,none (console)

    defaultpath = './models/pf_4_bicubic.pth'
    parser.add_argument('-m'  ,'--model',default=defaultpath, help='Type model path')

    parser.add_argument('-i'  ,'--input'   ,default='./sample/pds1.jpg'  ,metavar='{...}'    ,help='image path')
    parser.add_argument('-o'  ,'--out'  ,default= 'same' , help='where to save.')
    
    args = parser.parse_args()

    if args.out == 'same':
        in_filename = os.path.basename(os.path.splitext(args.input)[0] )
        args.out = os.path.join( os.path.dirname(args.input ) , f'pre_{in_filename}.jpg' )

    if args.model == defaultpath :
        model = get_instance_segmentation_model(num_classes,pretrained=False,mode='bicubic')
        model.to(device)
        m_path = args.model
        bicubic_pter = torch.load(m_path)
        model.load_state_dict(bicubic_pter['model_state_dict'])
        model.eval()
    
    # inference
    from PIL import Image
    from torchvision.transforms import functional as vfc 
    img     = Image.open(args.input).convert('RGB')

    imgt    = vfc.to_tensor(img).to(device = device)
    pre     = model([imgt])

    # draw prediction annotation (Mask ,Bbox ,score)
    annimg = draw_ann(img,pre[0]['scores'],pre[0]['boxes'],pre[0]['masks'].cpu(),nms=True,conf_threshold=0.4)
    annimg.save(args.out)
    