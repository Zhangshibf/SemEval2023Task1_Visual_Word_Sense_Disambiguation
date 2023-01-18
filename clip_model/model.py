from torch import nn
from transformers import CLIPVisionModelWithProjection,CLIPTokenizer, CLIPTextModelWithProjection

class clip_model(nn.Module):
    def __init__(self):
        super(clip_model, self).__init__()
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, text, image,setting):
        setting_types = ["text","image"]
        if setting not in setting_types:
            raise ValueError("Invalid data type. Expected one of: %s" % setting_types)

        if setting == "text":
            text_outputs = self.text_encoder(text)
            return text_outputs

        elif setting == "image":
            image_outputs = self.image_encoder(image)
            return image_outputs

class simple_nn(nn.Module):
    def __init__(self):
        super(simple_nn, self).__init__()
        self.text_layer = nn.Linear(512,300)
        self.image_layer = nn.Linear(512,300)

    def forward(self, text, image,setting):
        setting_types = ["text","image"]
        if setting not in setting_types:
            raise ValueError("Invalid data type. Expected one of: %s" % setting_types)

        if setting == "text":
            text_outputs = self.text_layer(text)
            return text_outputs

        elif setting == "image":
            image_outputs = self.image_layer(image)
            return image_outputs