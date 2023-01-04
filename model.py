#fine tune CLIP model
import os
import torch
from load_data import *
from torch.utils.data import Dataset, DataLoader,random_split
import requests
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel,CLIPVisionModel
from torch import nn
from transformers import CLIPProcessor, CLIPVisionModelWithProjection,CLIPTokenizer, CLIPTextModelWithProjection
import pandas as pd
from nltk.corpus import wordnet as wn
import nltk
from sentence_transformers import SentenceTransformer, util
import torch
from math import log
from torch import optim



#there is something wrong with the structure of this model. I need to fix this.
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
#            image_emd1 = image_outputs.last_hidden_state
#            image_emd2 = image_outputs.pooler_output
#            return image_emd1,image_emd2


def train_one_epoch(model,device,dataloader,optimizer):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
#    model = model.to(device)
    loss = 0
    criterion = ContrastiveLoss()
    # Train CLIP model for one epoch
    for keywords,contexts,augmentations,image_names,image_paths in dataloader:

        #generate embeddings for context + augmentation
        context_augemnted = list()
        for i,j in zip(contexts,augmentations):
            context_augemnted.append((i+" "+j))
        text_emds = list()

        for text in context_augemnted:
            # Tokenize the input text
            input_ids = tokenizer.encode(text)
            input_tensor = torch.tensor([input_ids])

            outputs = model(input_tensor,None,setting = "text")
            text_emds.append(outputs.text_embeds)

#            text_emd1,text_emd2 = model(input_tensor,None,setting = "text")
#            text_emds.append(text_emd2)

        image_emds = list()
        for i in image_paths:
            paths = i.split("#")
            images = open_images(paths)
            for k in images:
                outputs = model(None,k['pixel_values'],setting = "image")
                image_emds.append(outputs.image_embeds)

        # Compute the loss

#        loss_per_batch = compute_FLYP_loss(text_emds,image_emds)
        image_emds = torch.stack((image_emds))
        text_emds = torch.stack((text_emds))
        loss_per_batch = criterion(text_emds,image_emds)
        loss+=loss_per_batch
        model.zero_grad()
        # Backpropagate the loss and update the model weights
        loss_per_batch.backward()
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
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    transform = transforms.Compose(
        [transforms.Resize([1440, 1810]), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    images = list()
    for path in image_paths:
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert('RGB')
        image = transform(image)
        image = processor(images=image, return_tensors="pt")
        images.append(image)

    return images
class compute_FLYP_loss(nn.Module):
    def __init__(self, m=2.0):
        super(compute_FLYP_loss, self).__init__()
        self.m = m  # margin or radius

    def forward(self, y1, y2, d=0):
        euc_dist = nn.functional.pairwise_distance(y1, y2)

        if d == 0:
            return torch.mean(torch.pow(euc_dist, 2))  # distance squared
        else:  # d == 1
            delta = self.m - euc_dist  # sort of reverse distance
            delta = torch.clamp(delta, min=0.0, max=None)
            return torch.mean(torch.pow(delta, 2))  # mean over all rows

import torch
from torch import nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2):
        # Calculate pairwise distances between all image and text embeddings
        distances = torch.cdist(output1, output2, p=2)  # shape: (num_images, num_texts)

        # Calculate loss for each image embedding
        image_losses = []
        for i in range(output1.shape[0]):
            # Calculate distance between matching text embedding and image embedding
            pos_distance = torch.norm(output2[torch.argmin(distances[i])] - output1[i], p=2, dim=0)

            # Calculate distances between image embedding and all other text embeddings
            neg_distances = torch.norm(output2 - output1[i], p=2, dim=1)
            neg_distances[torch.argmin(distances[i])] = float("inf")  # exclude matching text embedding

            # Calculate loss for this image embedding
            loss = torch.mean(torch.max(torch.zeros_like(neg_distances), self.margin - pos_distance + neg_distances))
            image_losses.append(loss)

        # Calculate loss for each text embedding
        text_losses = []
        for i in range(output2.shape[0]):
            # Calculate distance between matching image embedding and text embedding
            pos_distance = torch.norm(output1[torch.argmin(distances[:, i])] - output2[i], p=2, dim=0)

            # Calculate distances between text embedding and all other image embeddings
            neg_distances = torch.norm(output1 - output2[i], p=2, dim=1)
            neg_distances[torch.argmin(distances[:, i])] = float("inf")  # exclude matching image embedding

            # Calculate loss for this text embedding
            loss = torch.mean(torch.max(torch.zeros_like(neg_distances), self.margin - pos_distance + neg_distances))
            text_losses.append(loss)

        # Return average loss across all image and text embeddings
        return (torch.mean(torch.tensor(image_losses)) + torch.mean(torch.tensor(text_losses))) / 2

"""
def compute_FLYP_loss(text_emds,image_emds):

    # Compute distance between text embedding and corresponding image embedding
    distances = list()
    total_loss = 0
    print(len(image_emds))
    print(image_emds[0].size())
    image_emds = torch.stack((image_emds))
    print(image_emds.size())

    for text_emd in text_emds:
        distances.append(torch.nn.functional.pairwise_distance(text_emd, image_emds))
    for i in range(len(text_emds)):
        text_images_distance = sum(distances[i])
        image_texts_distance = sum(list(k[i] for k in distances))
        similarity = distances[i][i]
        loss_per_pair = -log(similarity/text_images_distance)-log(similarity/image_texts_distance)
        total_loss+=loss_per_pair

    loss = total_loss/len(text_emds)

    return loss



class compute_FLYP_loss(nn.Module):
    def __init__(self, m=2.0):
        super(compute_FLYP_loss, self).__init__()
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
"""
def train_model(model,device,epoch,path_train,path_out,batch_size = 256):
    #train CLIP model for several epoches
    model.train()
    # Create the dataset
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

    for i in range(epoch):
        print("--------------Training Epoch {}---------------".format(i))
        avg_loss = train_one_epoch(model, device,train_dataloader, optimizer)
        print("--------------Loss per instance{}---------------".format(avg_loss))

        print("--------------Evaluation On Dev---------------")
        accuracy = evaluate(model, dev_dataloader)
        print("--------------Accuracy {}---------------".format(accuracy))

    print("--------------Final Evaluation On Test---------------")
    accuracy = evaluate(model, test_dataloader)
    print("--------------Accuracy {}---------------".format(accuracy))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument('--train', help="path to the train set")
    args = parser.parse_args()

    # Create the dataset
#    dataset = ImageTextDataset(args.train, data_type="train",device = device, text_augmentation=True)
    # Create the dataloader
#    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    model = clip_model()
#    model = model.to(device)
#    dataloader = dataloader.to(device)
    train_model(model, device = 'cuda' if torch.cuda.is_available() else 'cpu',epoch = 5, path_train=args.train, path_out="aa", batch_size=256)