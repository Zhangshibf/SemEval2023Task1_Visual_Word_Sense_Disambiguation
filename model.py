#contrastive learning
#models: ViLT, FLAVA, CLIP
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
import torchvision.transforms as transforms
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

        return text_emd1, text_emd2, image_emd1, image_emd2


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

def train_one_epoch(model,dataloader,text_augmentation=False,loss="FLYP"):
    # Train CLIP model for one epoch
    model.train()

    for text_image, labels in dataloader:
        # Apply text augmentation if specified
        if text_augmentation:
            input_data = apply_text_augmentation(input_data)

        # Make predictions with the model
        output = model(input_data)

        # Compute the loss
        if loss == "FLYP":
            loss = compute_FLYP_loss(output, labels)
        else:
            loss = compute_other_loss(output, labels)

        # Backpropagate the loss and update the model weights
        loss.backward()
        optimizer.step()

        return avg_loss, accuracy

def train_model(model,epoch,path_train,path_out,batch_size = 256,,text_augmentation=True,loss="FLYP"):
    #train CLIP model for several epoches
    # Create a transform to preprocess the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Create the dataset
    dataset = ImageTextDataset(path_train, transform=transform)
    # Create the dataloader
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    for i in epoch:
        print("--------------Epoch {}---------------".format(i))
        avg_loss, accuracy = train_one_epoch(model, dataloader=dataloader, text_augmentation=text_augmentation, loss=loss)
        print("--------------Loss per instance{}---------------".format(avg_loss))
        print("--------------Accuracy {}---------------".format(accuracy))

