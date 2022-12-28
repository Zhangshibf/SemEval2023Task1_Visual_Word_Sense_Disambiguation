#models: ViLT, FLAVA, CLIP
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
import torch


class clip_model():
    def __int__(self):
        super(clip_model, self).__init__()
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, text, image,setting):
        setting_types = ["text","image"]
        if setting not in setting_types:
            raise ValueError("Invalid data type. Expected one of: %s" % setting_types)

        if setting == "text":
            text_outputs = self.text_encoder(text)
            text_emd1 = text_outputs.last_hidden_state
            text_emd2 = text_outputs.pooler_output
            return text_emd1,text_emd2

        elif setting == "image":
            # encode image
            image_outputs = self.image_encoder(image)
            image_emd1 = image_outputs.last_hidden_state
            image_emd2 = image_outputs.pooler_output
            return image_emd1,image_emd2


def train_one_epoch(model,dataloader,optimizer,loss="FLYP"):
    # Train CLIP model for one epoch
    for keywords,contexts,augmentations,images,image_names in dataloader:
        #generate embeddings for context + augmentation
        context_augemnted = list()
        for i,j in zip(contexts,augmentations):
            context_augemnted.append((i+" "+j))
        text_emds = list()
        image_emds = list()

        for text in context_augemnted:
            text_emd1,text_emd2 = model(text,emb_type = "text")
            text_emds.append(text_emd2)

        for image in images:
            image_emd1,image_emd2 = model(image,emb_type = "image")
            image_emds.append(image_emd2)

        #for a text-image pair.
        #The negative samples for the text are positive sample of other texts in the same bacth + the nine negative sample given by dataset
        #the negative samples for the image are the other texts in the same batch.
        # Compute the loss
        if loss == "FLYP":
            #calculate the similarity score between text_emd and images of the same batch
            similarity_score = "something"
            loss = compute_FLYP_loss(similarity_scores)
        else:
            #I don't know what are other options... maybe with a linear layer?
            pass

        model.zero_grad()
        # Backpropagate the loss and update the model weights
        loss.backward()
        optimizer.step()

        return avg_loss, accuracy

def compute_FLYP_loss(text_embedding, image_embeddings, margin=0.1):
    # Compute distance between text embedding and corresponding image embedding
    positive_distance = torch.nn.functional.pairwise_distance(text_embedding, image_embeddings[0])

    # Compute distances between text embedding and negative image embeddings
    negative_distances = []
    for image_embedding in image_embeddings[1:]:
        negative_distance = torch.nn.functional.pairwise_distance(text_embedding, image_embedding)
        negative_distances.append(negative_distance)

    # Compute maximum negative distance
    max_negative_distance = torch.max(torch.stack(negative_distances))

    # Compute loss
    loss = torch.max(torch.tensor(0.), margin + positive_distance - max_negative_distance)

    return loss

def evaluate(model, dataloader):
    model.eval()

        

def train_model(model,epoch,path_train,path_out,batch_size = 256,text_augmentation=True,loss="FLYP"):
    #train CLIP model for several epoches
    model.train()
    # Create the dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = ImageTextDataset(path_train, data_type="train",device = device, text_augmentation=True)

    # Split the dataloader into train, dev, and test sets
    train_size = int(0.8 * len(dataset))
    dev_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - dev_size

    train_dataset, dev_dataset, test_dataset = random_split(dataset, [train_size, dev_size, test_size])

    # Create dataloaders for each set
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    optimizer = "something"

    for i in epoch:
        print("--------------Training Epoch {}---------------".format(i))
        avg_loss, accuracy = train_one_epoch(model, train_dataloader, optimizer,loss="FLYP")
        print("--------------Loss per instance{}---------------".format(avg_loss))
        print("--------------Accuracy {}---------------".format(accuracy))

        print("--------------Evaluation On Dev---------------")
        avg_loss, accuracy = evaluate(model, dev_dataloader)
        print("--------------Loss per instance{}---------------".format(avg_loss))
        print("--------------Accuracy {}---------------".format(accuracy))

