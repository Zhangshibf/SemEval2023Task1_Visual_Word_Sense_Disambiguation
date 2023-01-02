from PIL import Image
import requests
import os
import torch
from load_data import *
from torch.utils.data import Dataset, DataLoader,random_split
import requests
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel,CLIPVisionModel
from torch import nn
import pandas as pd

import torch
from math import log
from torch import optim
from transformers import CLIPProcessor, CLIPVisionModel

class clip_model(nn.Module):
    def __init__(self):
        super(clip_model, self).__init__()
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    def forward(self, text, image,setting):
        setting_types = ["text","image"]
        if setting not in setting_types:
            raise ValueError("Invalid data type. Expected one of: %s" % setting_types)

        if setting == "text":
            text_outputs = self.text_encoder(text)
            text_emd1 = text_outputs.last_hidden_state
            text_emd2 = text_outputs.pooler_output
            return text_emd1,text_emd2

        elif setting == "image":
            # encode image
            image_outputs = self.image_encoder(image)
            image_emd1 = image_outputs.last_hidden_state
            image_emd2 = image_outputs.pooler_output
            return image_emd1,image_emd2
#model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = clip_model()

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#image = Image.open(requests.get(url, stream=True).raw)
#print(type(image))
#<class 'PIL.JpegImagePlugin.JpegImageFile'>

image_paths = ['/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.86.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.155.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.68.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.9.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.72.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.158.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.7.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.132.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.36.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.27.jpg']
images = list()
ImageFile.LOAD_TRUNCATED_IMAGES = True
transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Resize([1440, 1810]),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])
for path in image_paths:
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert('RGB')
        print(type(image))
    image = transform(image)
    images.append(image)
inputs = processor(images=images, return_tensors="pt")
outputs = model(None,**inputs,"image")
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output  # pooled CLS states

print(pooled_output.shape)
