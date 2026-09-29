import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, datasets

import sys
import os
import scipy.io

import json
import PIL
import numpy as np
import matplotlib.pyplot as plt

import requests
url = "https://analytics.drake.edu/~reza/teaching/cs181_fall26/datasets/imagenet_1000_classes.json"

response = requests.get(url)
response.raise_for_status()
label_map = response.json()
print(label_map)

def display_prediction(output_prob, label_map, how_many=10):
    print(f"label map: {label_map}")
    '''
    topk          = torch.topk(output_prob, k=how_many)
    topk_prob     = topk.values[0]
    topk_indices  = topk.indices[0]
    '''

    topk            = torch.topk(output_prob, k=how_many, dim=1)
    topk_prob       = topk.values[0]
    topk_indices    = topk.indices[0]


    if device == 'cuda':
        topk_indices  = topk_indices.data.cpu().numpy()
    else:
        topk_indices  = topk_indices.data.numpy()

    print(f"top{how_many}_classes are as follow: \n{topk_indices}")
    labels        = "{:75s}".format("predicted label")
    prob          = "{}".format("probability")
    print("\n{:75s}".format("--------------------------------------------------------------------------------------\n"), labels, prob,"\n{:75s}".format("--------------------------------------------------------------------------------------"))

    for i in range(how_many):
        labels  = label_map[str(topk_indices[i])]
        labels  = "{:75s}".format(labels)
        prob    = "{:.4f}".format(float(topk_prob[i]))
        print(labels, prob)

    print("{:75s}".format("--------------------------------------------------------------------------------------"))


def get_imagenet_mean_std_normalized():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return mean, std

TEST_IMAGE_DIR  = '/content/drive/MyDrive/cs181_fall26/module_3_classification/images/'
image_name_list = ['random_dog1.jpg',
                   'random_dog2.jpg',
                   'random_cat1.jpg',
                   'random_cat2.jpg',
                   'random_bike1.jpg'
                   ]

# ------------------------Make a class the load pretrained weights of the transformer ViT------------------------------------------------
num_of_classes = 1000
model = ViT(num_of_classes)
model.to(device)

# ------------------------Read an image using PIL------------------------------------------------
img_index       = 3
img_name        = TEST_IMAGE_DIR + '/' + image_name_list[img_index]
img = PIL.Image.open(img_name)
plt.imshow(img)
plt.title("random image")

# ------------------------Transform the image by resizing + normalizing -------------------------
imagenet_mean, imagenet_std = get_imagenet_mean_std_normalized()
my_transform = transforms.Compose(
    [ transforms.Resize( (224, 224) ),
      transforms.ToTensor(),
      transforms.Normalize(mean=imagenet_mean, std=imagenet_std)]
)
img_transformed = my_transform(img).unsqueeze(0).to(device)
print(f"input image.shape={img_transformed.shape}")

# ------------------------Apply forward pass through your Transformer-----------------------------------------
model.eval()
raw_output = model(img_transformed)
output_prob = torch.softmax( raw_output, dim=1 )
display_prediction(output_prob, label_map, how_many=7)