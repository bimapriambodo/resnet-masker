#!/usr/bin/env python
# coding: utf-8


#infrence code to detect font styles.



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


filepath = r"C:\Users\aiforesee\Documents\GitHub\observations\mask_classifier\mask1_model.pth"
model = torch.load(filepath)


class_names = ['with_mask',
 'without_mask'
]



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    #pil_image = image
   
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = image_transforms(pil_image)
    return img




def classify_face(image_path):
    device = torch.device("cpu")
    img = process_image(image_path)
    print('image_processed')
    img = img.unsqueeze_(0)
    img = img.float()

    model.eval()
    model.cpu()
    output = model(img)
    print(output,'##############output###########')
    _, predicted = torch.max(output, 1)
    print(predicted.data[0],"predicted")


    classification1 = predicted.data[0]
    index = int(classification1)
    print(class_names[index])
    return class_names[index]

if __name__ == '__main__':
    map_location=torch.device('cpu')
    image_path = r"C:\Users\aiforesee\Documents\GitHub\observations\experiements\dest_folder\val\with_mask\469-with-mask.jpg"
    label = classify_face(image_path)
    print("the label is", label)









