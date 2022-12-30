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
from math import log
from torch import optim


class clip_model(nn.Module):
    def __init__(self):
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
    loss = 0
    # Train CLIP model for one epoch
    for keywords,contexts,augmentations,image_paths,image_names,negative_images_paths,negative_image_names in dataloader:
        #generate embeddings for context + augmentation
        context_augemnted = list()
        for i,j in zip(contexts,augmentations):
            context_augemnted.append((i+" "+j))
        text_emds = list()
        positive_image_emds = list()
        neg_image_emds = list()

        for text in context_augemnted:
            text_emd1,text_emd2 = model(text,None,setting = "text")
            text_emds.append(text_emd2)

        #positive images
        images = open_images(image_paths)
        for image in images:
            image_emd1,image_emd2 = model(None,image,setting = "image")
            positive_image_emds.append(image_emd2)

        #negative images

        for paths in negative_image_paths:
            temporary = list()
            neg_image = open_images(paths)
            for image in neg_image:
                image_emd1, image_emd2 = model(None, image, emb_type="image")
                temporary.append(image_emd2)
            neg_image_emds.append(temporary)

        # Compute the loss
        if loss == "FLYP":
            loss_per_batch = compute_FLYP_loss(text_emds,positive_image_emds,neg_image_emds)
        else:
            #I don't know what are other options... maybe with a linear layer?
            pass
        loss+=loss_per_batch
        model.zero_grad()
        # Backpropagate the loss and update the model weights
        loss.backward()
        optimizer.step()

    return loss

def evaluate(model, dataloader):
    model.eval()
    for keywords,contexts,augmentations,images,image_names,negative_images,negative_image_names in dataloader:
        #generate embeddings for context + augmentation
        context_augemnted = list()
        for i,j in zip(contexts,augmentations):
            context_augemnted.append((i+" "+j))
        text_emds = list()
        image_emds = list()
        positive_image_emds = list()
        neg_image_emds = list()

        for text in context_augemnted:
            text_emd1,text_emd2 = model(text,None,emb_type = "text")
            text_emds.append(text_emd2)

        for p_i,n_is in zip(images,negative_images):
            temporary = list()
            temporary.append(p_i)
            temporary.extend(n_is)
            temporary_emds = open_images(temporary)
            image_emds.append(temporary_emds)
        #calculate similarity, determine prediction
        total_similarities = list()

        for idx in range(len(image_emds[0])):
            column = [i[idx] for i in image_emds]
            similarities = torch.nn.functional.pairwise_distance(text_emds, column)
            total_similarities.append(similarities)
            prediction = np.argmax(total_similarities,axis=0)

        correct_prediction = 0
        for i in prediction:
            if i == 0:
                correct_prediction+=1

        accuracy = correct_prediction/len(prediction)

        return accuracy



def open_images(image_paths):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    images = list()
    for path in image_paths:
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert('RGB')
        images.append(image)
    return images

def compute_FLYP_loss(text_emds,p_image_emds,n_image_emds, margin=0.1):
    # Compute distance between text embedding and corresponding image embedding
    distances = list()
    for text_emd in text_emds:
        distances.append(torch.nn.functional.pairwise_distance(text_emd, p_image_emds))
    total_loss = 0
    for i in range(len(text_emds)):
        text_images_distance = sum(distances[i])
        image_texts_distance = sum(list(k[i] for k in distances))
        similarity = distances[i][i]
        loss_per_pair = -log(similarity/text_images_distance)-log(similarity/image_texts_distance)
        total_loss+=loss_per_pair

    loss = total_loss/len(text_emds)

    return loss

        

def train_model(model,epoch,path_train,path_out,batch_size = 256,loss="FLYP"):
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#    optimizer = optim.SGD(list(model.text_encoder.parameters()) + list(model.image_encoder.parameters()), lr=0.001,
#                          momentum=0.9)



    for i in range(epoch):
        print("--------------Training Epoch {}---------------".format(i))
        avg_loss = train_one_epoch(model, train_dataloader, optimizer,loss=loss)
        print("--------------Loss per instance{}---------------".format(avg_loss))
        print("--------------Accuracy {}---------------".format(accuracy))

        print("--------------Evaluation On Dev---------------")
        accuracy = evaluate(model, dev_dataloader)
        print("--------------Loss per instance{}---------------".format(avg_loss))
        print("--------------Accuracy {}---------------".format(accuracy))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument('--train', help="path to the train set")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Create the dataset
    dataset = ImageTextDataset(args.train, data_type="train",device = device, text_augmentation=True)
    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    model = clip_model()
    train_model(model, epoch = 5, path_train=args.train, path_out="aa", batch_size=256, loss="FLYP")

