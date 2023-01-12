#fine tune CLIP model
from load_data import *
from PIL import Image
import argparse
from PIL import ImageFile
import torchvision.transforms as transforms
from torch import nn
from transformers import CLIPProcessor, CLIPVisionModelWithProjection,CLIPTokenizer, CLIPTextModelWithProjection
import torch
from torch import optim
import clip

class clip_model(nn.Module):
    def __init__(self):
        super(clip_model, self).__init__()
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.linear1 = nn.Linear(1024,300)
        self.linear2 = nn.Linear(300,2)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, text, image):
        text_outputs = self.text_encoder(text).text_embeds
        image_outputs = self.image_encoder(image).image_embeds
        concat = torch.cat((text_outputs,image_outputs), 1)
        out1 = self.linear1(concat)
        out2 = self.linear2(out1)
        prediction = self.softmax(out2)

        return prediction


model = clip_model()
params = model.state_dict()
#active_param = ['linear1.weight', 'linear1.bias', 'linear2.weight', 'linear2.bias']
#freeze all params that are not 'linear1.weight', 'linear1.bias', 'linear2.weight', 'linear2.bias'
#for i,j in params.items():
#    if i not in active_param:
#        j.requires_grad = False

#for i in model.parameters():
#    print(i)

model.text_encoder.requires_grad_(False)
model.image_encoder.requires_grad_(False)
