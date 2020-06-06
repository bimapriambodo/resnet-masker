#!/usr/bin/env python
# coding: utf-8




import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import time
import copy
from PIL import Image
import glob
import cv2


class detection_image:

    def __init__(self):
        self.filepath = r"C:\Users\aiforesee\Google Drive (bimapriambodowr@gmail.com)\Digital Rise Indonesia\Object Detection\Masker Detection - Resnet\mask_classifier\mask4_model.pth"
        self.model = torch.load(self.filepath)
        self.class_names = ['with_mask','without_mask']
        self.pil_image = None
        self.image_transforms = None
        self.img = None
        self.device = None
        self.image = None
        self.im = None
        self.output = None
        self.predicted = None
        self.classification1 = None
        self.index = None

    def process_image(self, image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        #TODO: Process a PIL image for use in a PyTorch model
        #pil_image = Image.open(image)
        self.pil_image = self.image
    
        self.image_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.img = self.image_transforms(self.pil_image)
        return self.img
        
        


    def classify_face(self, image):
        self.device = torch.device("cuda")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        #im_pil = image.fromarray(image)
        #image = np.asarray(im)
        self.im = Image.fromarray(self.image)
        self.image = process_image(self.im)
        print('image_processed')
        self.img = self.image.unsqueeze_(0)
        self.img = self.image.float()

        self.model.eval()
        # model.cpu()
        self.model.cpu()
        self.output = self.model(self.image)
        print(self.output,'##############output###########')
        _, self.predicted = torch.max(self.output, 1)
        print(self.predicted.data[0],"predicted")

        self.classification1 = self.predicted.data[0]
        self.index = int(self.classification1)
        print(self.class_names[self.index])
        return self.class_names[self.index]


if __name__ == '__main__':
    #map_location=torch.device('cpu')
    image = cv2.imread('praj.jpg')
    label = classify_face(image)
    print("the label is", label)









