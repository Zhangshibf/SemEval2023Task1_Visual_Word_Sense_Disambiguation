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
import pandas as pd

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
    def __init__(self, data_dir, data_type,
                 transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])):
        types = ["inaturalist","train"]
        if data_type not in types:
            raise ValueError("Invalid data type. Expected one of: %s" % data_type)
        self.data_dir = data_dir
        self.transform = transform
        self.text = list()
        self.image_path = list()

        if data_type == "inaturalist":
            #I will write this part later
            pass
        elif data_type == "train":
            #this is for the original train set of the task
            #train.data.v1.txt, train.gold.v1.txt
            train_data= pd.read_csv(os.path.join(data_dir, "train.data.v1.txt"), sep="\t", header=None)
            label_data = pd.read_csv(os.path.join(data_dir, "train.gold.v1.txt"), sep="\t", header=None)
            keywords = list(train_data[0])
            contexts = list(train_data[1])
            self.text = contexts
            image_filenames = list(label_data[0])
            for filename in image_filenames:
                self.image_path.append(os.path.join(data_dir, "train_images_v1",filename))
                
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # Load the image and text
        image = Image.open(self.image_path[idx])
        text = self.texts[idx]

        if self.transform:
            image = self.transform(image)

        return image, text


#transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#])


def train_one_epoch(model,dataloader,optimizer,text_augmentation=False,loss="FLYP"):
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

def train_model(model,epoch,path_train,path_out,batch_size = 256,text_augmentation=True,loss="FLYP"):
    #train CLIP model for several epoches

    # Create the dataset
    dataset = ImageTextDataset(path_train,
                               data_type="train")
    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i in epoch:
        print("--------------Epoch {}---------------".format(i))
        avg_loss, accuracy = train_one_epoch(model, dataloader=dataloader, text_augmentation=text_augmentation, loss=loss)
        print("--------------Loss per instance{}---------------".format(avg_loss))
        print("--------------Accuracy {}---------------".format(accuracy))

