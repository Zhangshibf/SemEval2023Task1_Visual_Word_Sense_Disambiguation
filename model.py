#fine tune CLIP model
import os
import torch
from load_data import *
from torch.utils.data import Dataset, DataLoader,random_split
import requests
import torchvision.transforms as transforms
from torch import nn
from transformers import CLIPTextConfig,CLIPProcessor, CLIPVisionModelWithProjection,CLIPTokenizer, CLIPTextModelWithProjection
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
        configuration = CLIPTextConfig(max_position_embeddings=2048)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32",config= configuration,ignore_mismatched_sizes=True)
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

        image_emds = list()
        paths = [i.split("#")[0] for i in image_paths]
        images = open_images(paths)
        for k in images:
            outputs = model(None, k['pixel_values'], setting="image")
            image_emds.append(outputs.image_embeds)

        image_emds = torch.stack((image_emds)).squeeze(dim=1)
        text_emds = torch.stack((text_emds)).squeeze(dim=1)
        loss_per_batch = criterion(text_emds,image_emds)
        loss+=loss_per_batch
        model.zero_grad()

        # Backpropagate the loss and update the model weights
        loss_per_batch.backward()
        optimizer.step()

    return loss

def evaluate(model, dataloader):
    model.eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
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

        image_emds = list()
        paths = [i.split("#")for i in image_paths]
        for ps in paths:
            images = open_images(ps)
            temporary = list()
            for k in images:
                outputs = model(None, k['pixel_values'], setting="image")
                temporary.append(outputs.image_embeds)
            image_emds.append(temporary)

        #calculate similarity, determine prediction
        total_similarities = list()

        for idx in range(len(image_emds)):
            ten_images = torch.stack((image_emds[idx])).squeeze()
            text = text_emds[idx].squeeze()
            similarities = torch.nn.functional.pairwise_distance(text, ten_images)
            similarities = similarities.detach().numpy()
            total_similarities.append(similarities)
        total_similarities = np.array(total_similarities)
        prediction = np.argmax(total_similarities,axis=1)

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


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0):
        super().__init__()
        self.margin = margin

    def forward(self, image_embeddings, text_embeddings):
        # calculate positive distance between matching image and text embeddings
        positive_distance = (image_embeddings - text_embeddings).pow(2).sum(1)
        # calculate negative distance between all other image and text embeddings
        negative_distance = torch.zeros(image_embeddings.size(0))
        for i in range(image_embeddings.size(0)):
            for j in range(image_embeddings.size(0)):
                if i != j:
                    negative_distance[i] += (image_embeddings[i] - text_embeddings[j]).pow(2).sum()
        negative_distance = negative_distance / (image_embeddings.size(0) - 1)
        # calculate loss
        loss = torch.mean((positive_distance - negative_distance + self.margin).clamp(min=0))
        return loss


def train_model(model,device,epoch,path_train,path_out,batch_size =256):
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