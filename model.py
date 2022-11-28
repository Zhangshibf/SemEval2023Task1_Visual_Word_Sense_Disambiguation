#contrastive learning
#models: ViLT, FLAVA, CLIP, data2vec
import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel,CLIPVisionModel
from torch import nn

class clip_model():
    def __int__(self):
        super(clip_model, self).__init__()
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.linear = nn.Linear(600,out_features = 2)
        self.softmax = nn.Softmax()
    def forward(self,image,text):
        #encode text
        text_outputs = self.text_encoder(text)
        text_emd1 = text_outputs.last_hidden_state
        text_emd2 = text_outputs.pooler_output

        #encode image
        image_outputs = self.image_encoder(image)
        image_emd1 = image_outputs.last_hidden_state
        image_emd2 = image_outputs.pooler_output

        concatenation = "some way to concatenate text embeddings and image embeddings"
        output_linear = self.linear(concatenation)
        prediction = self.softmax(output_linear)

        return prediction
