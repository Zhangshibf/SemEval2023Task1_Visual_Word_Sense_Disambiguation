import clip
import open_clip
import os
from load_data import *
import torch
from load_data import *
from torch.utils.data import Dataset, DataLoader, random_split
import requests
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPTokenizer, CLIPTextModel, CLIPVisionModel
from torch import nn
import pandas as pd
from nltk.corpus import wordnet as wn
import nltk
from sentence_transformers import SentenceTransformer, util
import torch
from math import log
from torch import optim
#fine tune CLIP model
import os
import torch
from load_data import *
from torch.utils.data import Dataset, DataLoader,random_split
import requests
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel,CLIPVisionModel
from torch import nn
import pandas as pd
from nltk.corpus import wordnet as wn
import nltk
from sentence_transformers import SentenceTransformer, util
import torch as T
from math import log
from torch import optim
class clip_model(nn.Module):
    def __init__(self):
        super(clip_model, self).__init__()
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
#        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
#        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    def forward(self, text, image,setting):
        setting_types = ["text","image"]
        if setting not in setting_types:
            raise ValueError("Invalid data type. Expected one of: %s" % setting_types)

        if setting == "text":
            text_outputs = self.text_encoder(text)
            return text_outputs
#            text_emd1 = text_outputs.last_hidden_state
#            text_emd2 = text_outputs.pooler_output
#            return text_emd1,text_emd2

        elif setting == "image":
            # encode image
            image_outputs = self.image_encoder(image)
            return image_outputs

class ContrastiveLoss(nn.Module):
    def __init__(self, m=2.0):
        super(ContrastiveLoss, self).__init__()
        self.m = m  # margin or radius

    def forward(self, y1, y2, d=0):
        # d = 0 means y1 and y2 are supposed to be same
        # d = 1 means y1 and y2 are supposed to be different

        euc_dist = nn.functional.pairwise_distance(y1, y2)

        if d == 0:
            return torch.mean(torch.pow(euc_dist, 2))  # distance squared
        else:  # d == 1
            delta = self.m - euc_dist  # sort of reverse distance
            delta = torch.clamp(delta, min=0.0, max=None)
            return torch.mean(torch.pow(delta, 2))  # mean over all rows

def main():
  print("\nBegin contrastive loss demo \n")

  loss_func = ContrastiveLoss()

  y1 = T.tensor([[1.0, 2.0, 3.0],
                 [3.0, 4.0, 5.0]], dtype=T.float32).to(device)

  y2 = T.tensor([[1.0, 2.0, 3.0],
                 [3.0, 4.0, 5.0]], dtype=T.float32).to(device)

  y3 = T.tensor([[10.0, 20.0, 30.0],
                 [30.0, 40.0, 50.0]], dtype=T.float32).to(device)

  loss = loss_func(y1, y2, 0)
  print(loss)  # 0.0 -- small; y1 y2 should be equal

  loss = loss_func(y1, y2, 1)
  print(loss)  # 4.0 -- large; y1 y2 should be different

  loss = loss_func(y1, y3, 0)
  print(loss)  # 2591.99 -- large; y1 y3 should be equal

  loss = loss_func(y1, y3, 1)
  print(loss)  # 0.0 -- small; y1 y2 should be different

  loss.backward()

  print("\nEnd demo ")

if __name__ == "__main__":
  main()