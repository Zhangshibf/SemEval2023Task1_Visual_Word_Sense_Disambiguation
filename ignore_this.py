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

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#image = Image.open(requests.get(url, stream=True).raw)

#inputs = processor(images=image, return_tensors="pt")
image_paths = ['/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.86.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.155.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.68.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.9.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.72.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.158.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.7.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.132.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.36.jpg', '/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.27.jpg']

ImageFile.LOAD_TRUNCATED_IMAGES = True
transform = transforms.Compose(
    [transforms.ToPILImage(),transforms.Resize([1440, 1810]), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])
for path in image_paths:
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert('RGB')
    image = transform(image)
    print(type(image))
#        image = image.unsqueeze(0)
    image = processor(images=image, return_tensors="pt")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    pooled_output = outputs.pooler_output  # pooled CLS states

    print(pooled_output.shape())
