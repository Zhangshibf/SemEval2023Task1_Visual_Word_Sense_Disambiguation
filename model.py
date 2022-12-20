#contrastive learning
#models: ViLT, FLAVA, CLIP
import os
import torch
from torch.utils.data import Dataset, DataLoader
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



class ImageTextDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_filenames = []
        self.texts = []

        # Load the image filenames and texts into the lists
        for file in os.listdir(data_dir):
            if file.endswith(".jpg"):
                self.image_filenames.append(os.path.join(data_dir, file))
            elif file.endswith(".txt"):
                with open(os.path.join(data_dir, file), "r") as f:
                    self.texts.append(f.read())

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load the image and text
        image = Image.open(self.image_filenames[idx])
        text = self.texts[idx]

        if self.transform:
            image = self.transform(image)

        return image, text

# Example usage:

# Create a transform to preprocess the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create the dataset
dataset = ImageTextDataset("/path/to/data", transform=transform)

# Create the dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# Iterate through the dataloader
for images, texts in dataloader:
    # Use the images and texts to train your model
    model.train(images, texts)
