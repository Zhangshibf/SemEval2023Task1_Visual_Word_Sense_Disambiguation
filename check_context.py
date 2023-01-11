from load_data import *
from PIL import Image
import argparse
from PIL import ImageFile
import torchvision.transforms as transforms
from torch import nn
from transformers import CLIPTextConfig,CLIPProcessor, CLIPVisionModelWithProjection,CLIPTokenizer, CLIPTextModelWithProjection
import torch
from torch import optim
class clip_model(nn.Module):
    def __init__(self):
        super(clip_model, self).__init__()
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, text, image,setting):
        if setting == "text":
            text_outputs = self.text_encoder(text)
            return text_outputs

        elif setting == "image":
            image_outputs = self.image_encoder(image)
            return image_outputs
        else:
            text_outputs = self.text_encoder(text)
            image_outputs = self.image_encoder(image)
            return text_outputs,image_outputs
def open_images(image_paths):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    transform = transforms.Compose(
        [transforms.Resize([1440, 1810]), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    images = list()
    for path in image_paths:
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert('RGB')
        image = transform(image)
        image = processor(images=image, return_tensors="pt")
        images.append(image)

    return images
model = clip_model()

text = "I had an ice cream today. It was nice!"
paths = ["/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/image.6331.jpg","/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/image.6331.jpg"]
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",model_max_length=77)
tokens = torch.tensor([tokenizer.encode(text,max_length=77,truncation=True)])
text_emds = model(tokens,None,setting = "text")

images = open_images(paths)
image_emds = list()
for k in images:
    input_image = k['pixel_values']
    input_image = input_image
    outputs = model(None, input_image, setting="image")
    image_emds.append(outputs.image_embeds)

text_emds_p, image_emds_p = model(tokens, images[0]['pixel_values'], setting="balbla")

print(text_emds)
print(text_emds_p)
