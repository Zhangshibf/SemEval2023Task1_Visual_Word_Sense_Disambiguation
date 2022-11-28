#contrastive learning
#models: ViLT, FLAVA, CLIP, data2vec
import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel

class lv_model():
    def __int__(self):
        super(lv_model, self).__init__()
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

#        self.clip= CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.linear = nn.Linear(,out_features = 10)
        self.softmax = nn.Softmax()
    def forward(self,image,text):
        pass







model =
