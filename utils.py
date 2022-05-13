# -*- coding: utf-8 -*-
"""Implements some utils

TODO:
"""

import random
from random import choice
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
#from myssim import compare_ssim as ssim
import cv2
import math
import torch
#from skimage import img_as_float
#from skimage.color import rgb2ycbcr
#from skimage.measure import compare_psnr, compare_ssim

def exp_lr_scheduler(optimizer, name):  ##https://discuss.pytorch.org/t/adaptive-learning-rate/320/26
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] /= 2
    print('{}-lr is set to {}'.format(name,param_group['lr']))

def output_psnr_mse(img_orig, img_out):
    img_orig = img_orig.astype(float)/255.0
    img_out = img_out.astype(float)/255.0
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

def compute_mssim(ref_img, res_img):
    channels = []
    c = 1 if len(res_img[2].shape) == 2 else 3
    for i in range(c):
        channels.append(ssim(ref_img[:,:,i],res_img[:,:,i],
        gaussian_weights=True, use_sample_covariance=False))
    return np.mean(channels)

def is_same_size(ref_img,res_img):  ##这个函数官方文件没有，为了解决生成的图像不匹配而改动的
    if (ref_img.shape) != (res_img.shape):   
        res_img =cv2.resize(res_img, dsize=(ref_img.shape[1],ref_img.shape[0]),interpolation=cv2.INTER_CUBIC)
    return ref_img,res_img

def mod_crop(im, scale):
    print(im.shape[:2])
    h, w = im.shape[:2]
    # return im[(h % scale):, (w % scale):, ...]
    return im[:h - (h % scale), :w - (w % scale), ...]

def crop_boundaries(im, cs):
    if cs > 1:
        return im[cs:-cs, cs:-cs, ...]
    else:
        return im

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    print(img.shape)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        C ,H, W= img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:,:H - H_r, :W - W_r]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    print("after:{}".format(img.shape))
    return img
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def eval_psnr_and_ssim(im1, im2, scale):
    im1_t = np.atleast_3d(img_as_float(im1))
    im2_t = np.atleast_3d(img_as_float(im2))

    if im1_t.shape[2] == 1 or im2_t.shape[2] == 1:
        im1_t = im1_t[..., 0]
        im2_t = im2_t[..., 0]

    else:
        im1_t = rgb2ycbcr(im1_t)[:, :, 0:1] / 255.0
        im2_t = rgb2ycbcr(im2_t)[:, :, 0:1] / 255.0

    if scale > 1:
        im1_t = mod_crop(im1_t, scale)
        im2_t = mod_crop(im2_t, scale)

        # NOTE conventionally, crop scale+6 pixels (EDSR, VDSR etc)
        im1_t = crop_boundaries(im1_t, int(scale) + 6)
        im2_t = crop_boundaries(im2_t, int(scale) + 6)

    psnr_val = compare_psnr(im1_t, im2_t)
    ssim_val = compare_ssim(
        im1_t,
        im2_t,
        win_size=11,
        gaussian_weights=True,
        multichannel=True,
        data_range=1.0,
        K1=0.01,
        K2=0.03,
        sigma=1.5)

    return psnr_val, ssim_val

def tensor2PIL(img):
    return transforms.ToPILImage()(img)

def RGB2YCbCr_Y(img):
    batch = img.shape[0]
    W = img.shape[2]
    H = img.shape[3]
    outputs = np.zeros((batch, 1, W, H))
    for i in range(batch):
        output = tensor2PIL(img[i])
        r, g, b = output.split()
        r = np.array(r)
        g = np.array(g)
        b = np.array(b)
        outputs[i] = 0.256789 * r + 0.504129 * g + 0.097906 * b + 16
    return outputs
def RGB2YCbCr_CbCr(img):
    batch = img.shape[0]
    W = img.shape[2]
    H = img.shape[3]
    Cb = np.zeros((batch, W, H))
    Cr = np.zeros((batch, W, H))
    for i in range(batch):
        output = tensor2PIL(img[i])
        r, g, b = output.split()
        r = np.array(r)
        g = np.array(g)
        b = np.array(b)
        Cb[i] = -0.148223 * r - 0.290992 * g + 0.439215 * b + 128
        Cr[i] = 0.439215 * r - 0.367789 * g - 0.071426 * b + 128
    return Cb,Cr

def YCbCr2RGB(Y,Cb,Cr):
    #print(Cb.shape)
    batch = Y.shape[0]
    H = Y.shape[2]
    W = Y.shape[3]
    img_rgb1 = np.zeros((batch, H, W,3))
    img_rgb2 = np.zeros((batch,3, H, W),dtype=np.float32)
    img_rgb2 = torch.tensor(img_rgb2).cuda()
    #print(Y.max(),Y.min())
    #print(Cb.max(), Cb.min())
    #print(Cr.max(), Cr.min())
    for i in range(batch):
        y = Y[i].detach().numpy()
        img_rgb1[i, :, :,0] = 1.164383 * (y - 16) + 1.596027 * (Cr[i] - 128)
        img_rgb1[i, :, :,1] = 1.164383 * (y - 16) - 0.391762 * (Cb[i] - 128) - 0.812969 * (Cr[i] - 128)
        img_rgb1[i, :, :,2] = 1.164383 * (y - 16) + 2.017230 * (Cb[i] - 128)
        #print(img_rgb1[i].max(),img_rgb1[i].min())
        img_rgb1[i][img_rgb1[i]<0] = 0    ##因为Y通道边缘减半甚至更少,使得出现负数,做个大胆的做法,把负数附为0
        img_rgb1 = np.round(img_rgb1).astype(np.uint8)
        img_rgb2[i] = transforms.ToTensor()(img_rgb1[i])
    #img_rgb = img_rgb.astype(np.uint8)

    #print(img_rgb2[3].max(),img_rgb2[3].min())

    return img_rgb2

def gradient(img):
    
    img = cv2.cvtColor(img,cv2.COLOR_RGB2YCR_CB)
    img=img[:,:,0]
    height = img.shape[1] -1
    width = img.shape[0] - 1

    tmp = []
    for i in range(width):
        for j in range(height):
            dx = (img[i][j+1]).astype(np.float) - (img[i][j]).astype(np.float)
            dy = (img[i+1][j]).astype(np.float) - (img[i][j]).astype(np.float)
            ds = math.sqrt((dx**2 + dy**2)/2.0)
            tmp.append(ds)
    
    mgrad = np.mean(tmp)
    return mgrad

    