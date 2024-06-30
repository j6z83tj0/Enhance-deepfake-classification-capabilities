import os
import cv2
import sys
import torch
import torch.nn
import argparse
import numpy as np
from PIL import Image
from PIL import ImageFile
from PIL import ImageFilter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from networks.resnet import resnet50
import time
from networks.fusing_resnet import fusing_resnet
import torchvision.transforms.functional as TF

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f','--file', default='examples_realfakedir')
parser.add_argument('-m','--model_path', type=str, default='CNNDetection-master/checkpoints/sdv14_fusingmodel_ycc_hsv_EDGE_ENHANCE/model_epoch_latest.pth')
parser.add_argument('-c','--crop', type=int, default=None, help='by default, do not crop. specify crop size')
parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')
parser.add_argument('--filter',type=str,default=None,help='None,"CONTOUR","DETAIL","EDGE_ENHANCE","EMBOSS","FIND_EDGES","SMOOTH","SHARPEN","UnsharpMask","ModeFilter","GAUSSIAN_BLUR"')
parser.add_argument('--rz_interp', default='bilinear')
parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
opt = parser.parse_args()

model = fusing_resnet()
state_dict = torch.load(opt.model_path, map_location='cpu')
model.load_state_dict(state_dict['model'])
if(not opt.use_cpu):
  model.cuda()
model.eval()

# Transform
def data_augment_ycc(img, opt):
    img = np.array(img)
    ycc_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if opt.filter == "GAUSSIAN_BLUR" :
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(ycc_image, sig)
        # method = sample_discrete(opt.jpg_method)
        # qual = sample_discrete(opt.jpg_qual)
        # ycc_image = jpeg_from_key(ycc_image, qual, method)
    else :    
    # additional augmentation  
        ycc_image =Image.fromarray(ycc_image )
        filter={"CONTOUR":ImageFilter.CONTOUR,
                "DETAIL":ImageFilter.DETAIL,
                "EDGE_ENHANCE":ImageFilter.EDGE_ENHANCE,
                "EMBOSS":ImageFilter.EMBOSS,
                "FIND_EDGES":ImageFilter.FIND_EDGES,
                "SMOOTH":ImageFilter.SMOOTH,
                "SHARPEN":ImageFilter.SHARPEN,
                "UnsharpMask":ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3),
                "ModeFilter":ImageFilter.ModeFilter(5)}
        if opt.filter!=None :
            ycc_image = ycc_image.filter(filter[opt.filter])
    
    # img.save(f'pic/{random()}.jpg')    
    
    return ycc_image

def data_augment_hsv(img, opt):
    img = np.array(img)
    img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    if opt.filter == "GAUSSIAN_BLUR" :
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img_hsv, sig)
        # method = sample_discrete(opt.jpg_method)
        # qual = sample_discrete(opt.jpg_qual)
        # img_hsv = jpeg_from_key(img_hsv, qual, method)
    else :   
        # additional augmentation  
        img_hsv=Image.fromarray(img_hsv)
        filter={"CONTOUR":ImageFilter.CONTOUR,
                "DETAIL":ImageFilter.DETAIL,
                "EDGE_ENHANCE":ImageFilter.EDGE_ENHANCE,
                "EMBOSS":ImageFilter.EMBOSS,
                "FIND_EDGES":ImageFilter.FIND_EDGES,
                "SMOOTH":ImageFilter.SMOOTH,
                "SHARPEN":ImageFilter.SHARPEN,
                "UnsharpMask":ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3),
                "ModeFilter":ImageFilter.ModeFilter(5)}
        if opt.filter!=None :
            img_hsv = img_hsv.filter(filter[opt.filter])
        
    return img_hsv

rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    return TF.resize(img, (opt.loadSize, opt.loadSize), interpolation=rz_dict[opt.rz_interp])


trans_init = []
if(opt.crop is not None):
  trans_init = [transforms.CenterCrop(opt.crop),]
  print('Cropping to [%i]'%opt.crop)
else:
  print('Not cropping')
trans_ycc = transforms.Compose(trans_init + [
    transforms.Lambda(lambda img: custom_resize(img, opt)),
    transforms.Lambda(lambda img: data_augment_ycc(img, opt)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trans_init = []
if(opt.crop is not None):
  trans_init = [transforms.CenterCrop(opt.crop),]
  print('Cropping to [%i]'%opt.crop)
else:
  print('Not cropping')
trans_hsv = transforms.Compose(trans_init + [
  transforms.Lambda(lambda img: custom_resize(img, opt)),
    transforms.Lambda(lambda img: data_augment_hsv(img, opt)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])




img_ycc = trans_ycc(Image.open(opt.file).convert('RGB'))
img_hsv = trans_hsv(Image.open(opt.file).convert('RGB'))
start_time=time.time()
with torch.no_grad():
    in_ycc_tens = img_ycc.unsqueeze(0)
    in_hsv_tens = img_hsv.unsqueeze(0)
    if(not opt.use_cpu):
      in_ycc_tens = in_ycc_tens.cuda()
      in_hsv_tens = in_hsv_tens.cuda()
    prob = model(in_ycc_tens, in_hsv_tens).sigmoid().item()
end_time=time.time()
latency = end_time-start_time
print(f"latency : {latency}")
print('probability of being synthetic: {:.2f}%'.format(prob * 100))
