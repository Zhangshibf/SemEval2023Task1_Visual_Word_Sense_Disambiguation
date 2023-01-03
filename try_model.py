import clip
import open_clip
import os
from load_data import *
import torch
from load_data import *
from torch.utils.data import Dataset, DataLoader, random_split
import requests
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPTokenizer, CLIPTextModel, CLIPVisionModel
from torch import nn
import pandas as pd
from nltk.corpus import wordnet as wn
import nltk
from sentence_transformers import SentenceTransformer, util
import torch
from math import log
from torch import optim
#fine tune CLIP model
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

class CLIPEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.train_preprocess, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32-quickgelu', pretrained='laion400m_e31')
    def forward(self, images, text):
        return self.model(images, text)

    def save(self, filename):
        print(f'Saving clip encoder to {filename}')
        utils.torch_save(self, filename)



def train_one_epoch(model, dataloader, optimizer, loss="FLYP"):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    loss = 0
    # Train CLIP model for one epoch
    for keywords, contexts, augmentations, image_names, image_paths in dataloader:
        # generate embeddings for context + augmentation
        context_augemnted = list()
        for i, j in zip(contexts, augmentations):
            context_augemnted.append((i + " " + j))
        text_emds = list()
        text = context_augemnted[0]
        image_ps = image_paths[0].split("#")
        images = open_images(model.preprocess,image_ps)
        image_f,text_f,logit = model(images,text)
        print(len(image_f))
        print("done")

"""        for text in context_augemnted:
            # Tokenize the input text
            input_ids = tokenizer.encode(text)
            # Convert the input_ids to a tensor
            input_tensor = torch.tensor([input_ids])
            text_emd1, text_emd2 = model(text = input_tensor)
            text_emds.append(text_emd2)

        image_emds = list()
        for i in image_paths:
            paths = i.split("#")
            images = open_images(model.preprocess,paths)
            image_emd,_,_ = model(images)
            image_emds.append(image_emd)
"""
        # Compute the loss
        if loss == "FLYP":
            loss_per_batch = compute_FLYP_loss(text_emds, positive_image_emds, neg_image_emds)
        else:
            # I don't know what are other options... maybe with a linear layer?
            pass
        loss += loss_per_batch
        model.zero_grad()
        # Backpropagate the loss and update the model weights
        loss.backward()
        optimizer.step()

    return loss


def evaluate(model, dataloader):
    model.eval()
    for keywords, contexts, augmentations, images, image_names, negative_images, negative_image_names in dataloader:
        # generate embeddings for context + augmentation
        context_augemnted = list()
        for i, j in zip(contexts, augmentations):
            context_augemnted.append((i + " " + j))
        text_emds = list()
        image_emds = list()

        for text in context_augemnted:
            text_emd1, text_emd2 = model(text, None, emb_type="text")
            text_emds.append(text_emd2)

        for p_i, n_is in zip(images, negative_images):
            temporary = list()
            temporary.append(p_i)
            temporary.extend(n_is)
            temporary_emds = open_images(temporary)
            image_emds.append(temporary_emds)
        # calculate similarity, determine prediction
        total_similarities = list()

        for idx in range(len(image_emds[0])):
            column = [i[idx] for i in image_emds]
            similarities = torch.nn.functional.pairwise_distance(text_emds, column)
            total_similarities.append(similarities)
            prediction = np.argmax(total_similarities, axis=0)

        correct_prediction = 0
        for i in prediction:
            if i == 0:
                correct_prediction += 1

        accuracy = correct_prediction / len(prediction)

        return accuracy


def open_images(preprocess,image_paths):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    transform = transforms.Compose(
        [transforms.Resize([1440, 1810]), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    images = list()
    for path in image_paths:
        image = Image.open(path).convert('RGB')
        images.append(preprocess(image))

    return images


def compute_FLYP_loss(text_emds, p_image_emds, n_image_emds, margin=0.1):
    # Compute distance between text embedding and corresponding image embedding
    distances = list()
    for text_emd in text_emds:
        distances.append(torch.nn.functional.pairwise_distance(text_emd, p_image_emds))
    total_loss = 0
    for i in range(len(text_emds)):
        text_images_distance = sum(distances[i])
        image_texts_distance = sum(list(k[i] for k in distances))
        similarity = distances[i][i]
        loss_per_pair = -log(similarity / text_images_distance) - log(similarity / image_texts_distance)
        total_loss += loss_per_pair

    loss = total_loss / len(text_emds)

    return loss


def train_model(model, epoch, path_train, path_out, batch_size=256, loss="FLYP"):
    # train CLIP model for several epoches
    model.train()
    # Create the dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = ImageTextDataset(path_train, data_type="train", device=device, text_augmentation=True)

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
        avg_loss = train_one_epoch(model, train_dataloader, optimizer, loss=loss)
        print("--------------Loss per instance{}---------------".format(avg_loss))
        print("--------------Accuracy {}---------------".format(accuracy))

        print("--------------Evaluation On Dev---------------")
        accuracy = evaluate(model, dev_dataloader)
        print("--------------Accuracy {}---------------".format(accuracy))

    print("--------------Final Evaluation On Test---------------")
    accuracy = evaluate(model, Test_dataloader)
    print("--------------Accuracy {}---------------".format(accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument('--train', help="path to the train set")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Create the dataset
    dataset = ImageTextDataset(args.train, data_type="train", device=device, text_augmentation=True)
    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    model = CLIPEncoder()
    train_model(model, epoch=5, path_train=args.train, path_out="aa", batch_size=256, loss="FLYP")