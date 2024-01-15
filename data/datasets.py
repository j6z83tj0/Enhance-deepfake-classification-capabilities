import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from PIL import ImageFilter
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        # return binary_dataset(opt, root)
        return DoublePreprocessedDataset(root,opt)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


# def binary_dataset(opt, root):
#     if opt.isTrain:
#         crop_func = transforms.RandomCrop(opt.cropSize)
#     elif opt.no_crop:
#         crop_func = transforms.Lambda(lambda img: img)
#     else:
#         crop_func = transforms.CenterCrop(opt.cropSize)

#     if opt.isTrain and not opt.no_flip:
#         flip_func = transforms.RandomHorizontalFlip()
#     else:
#         flip_func = transforms.Lambda(lambda img: img)
#     if not opt.isTrain and opt.no_resize:
#         rz_func = transforms.Lambda(lambda img: img)
#     else:
#         rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        
#     dset = datasets.ImageFolder(
#             root,
#             transforms.Compose([
#                 rz_func,
#                 transforms.Lambda(lambda img: data_augment(img, opt)),
#                 # crop_func,
#                 flip_func,
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ]))
#     return dset


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path
    
def data_augment_rgb(img, opt):
    img = np.array(img)
    if opt.filter == "GAUSSIAN_BLUR" :
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)
        # method = sample_discrete(opt.jpg_method)
        # qual = sample_discrete(opt.jpg_qual)
        # img = jpeg_from_key(img, qual, method)
    else :    
        # additional augmentation  
        img =Image.fromarray(img )
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
            img = img.filter(filter[opt.filter])
        
        # img.save(f'pic/{random()}.jpg')       
     
    return img 

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

def data_ycc(img):
    img = np.array(img)
    ycc_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycc_image=Image.fromarray(ycc_image)
    return ycc_image

def data_hsv(img):
    img = np.array(img)
    img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img_hsv=Image.fromarray(img_hsv)
    return img_hsv


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, (opt.loadSize, opt.loadSize), interpolation=rz_dict[interp])



class DoublePreprocessedDataset(Dataset):
    def __init__(self, root, opt):
        self.opt=opt
        self.dataset = datasets.ImageFolder(root)
        if self.opt.isTrain:
            crop_func = transforms.RandomCrop(self.opt.cropSize)
        elif self.opt.no_crop:
            crop_func = transforms.Lambda(lambda img: img)
        else:
            crop_func = transforms.CenterCrop(self.opt.cropSize)

        if self.opt.isTrain and not self.opt.no_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = transforms.Lambda(lambda img: img)
        if not self.opt.isTrain and self.opt.no_resize:
            rz_func = transforms.Lambda(lambda img: img)
        else:
            rz_func = transforms.Lambda(lambda img: custom_resize(img, self.opt))
        self.transform_YCC = transforms.Compose([rz_func,
                                              transforms.Lambda(lambda img: data_augment_ycc(img, self.opt)),
                                              transforms.ToTensor(),  
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),         
                                            ])
        self.transform_HSV = transforms.Compose([rz_func, 
                                              transforms.Lambda(lambda img: data_augment_hsv(img, self.opt)),
                                              transforms.ToTensor(),           
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ])
        self.transform_YCC_noaug = transforms.Compose([rz_func,
                                            transforms.Lambda(lambda img: data_ycc(img)),
                                              transforms.ToTensor(),  
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),         
                                            ])
        self.transform_HSV_noaug = transforms.Compose([rz_func, 
                                              transforms.Lambda(lambda img: data_hsv(img)),
                                              transforms.ToTensor(),           
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            ])
        

    def __getitem__(self, index):
        image, label = self.dataset[index]
        rand=random()
        if self.opt.eval_mode !=True :
            if rand < 0.1:
                image_RGB = self.transform_YCC(image)
                image_HSV = self.transform_HSV(image)
            else:
                image_RGB = self.transform_YCC_noaug(image)
                image_HSV = self.transform_HSV_noaug(image)
        else :
            image_RGB = self.transform_YCC(image)
            image_HSV = self.transform_HSV(image)
        return image_RGB, image_HSV, label

    def __len__(self):
        return len(self.dataset)

