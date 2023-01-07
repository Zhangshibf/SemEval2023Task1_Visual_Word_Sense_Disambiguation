#fine tune CLIP model

from load_data import *
from PIL import Image
import argparse
from PIL import ImageFile
import torchvision.transforms as transforms
from torch import nn
from transformers import CLIPTextConfig,CLIPProcessor, CLIPVisionModelWithProjection,CLIPTokenizer, CLIPTextModelWithProjection
import torch
from torch import optim

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
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",model_max_length=77)
    loss = 0
    criterion = ContrastiveLoss()
    # Train CLIP model for one epoch
    for keywords,contexts,augmentations,image_names,image_paths in dataloader:
        text_emds = list()
        tokens = list()
        for i, j in zip(contexts, augmentations):
            context_augmented = i + " " + j
            # Tokenize the input text
            input_ids = torch.tensor([tokenizer.encode(context_augmented,max_length=77,truncation=True)])
            tokens.append(input_ids)

        for t in tokens:
            t = t.to(device)
            outputs = model(t,None,setting = "text")
            text_emds.append(outputs.text_embeds)

        image_emds = list()
        paths = [i.split("#")[0] for i in image_paths]
        images = open_images(paths)
        for k in images:
            input_image = k['pixel_values']
            input_image = input_image.to(device)
            outputs = model(None, input_image, setting="image")
            image_emds.append(outputs.image_embeds)

        image_emds = torch.stack((image_emds)).squeeze(dim=1)
        text_emds = torch.stack((text_emds)).squeeze(dim=1)
        image_emds = image_emds.to(device)
        text_emds = text_emds.to(device)

        loss_per_batch = criterion(text_emds,image_emds,device)
        loss+=float(loss_per_batch)
        model.zero_grad()

        # Backpropagate the loss and update the model weights
        loss_per_batch.backward()
        optimizer.step()

    return loss

def evaluate(model,device, dataloader):
    model.eval()
    correct = 0
    total = 0
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",model_max_length=77)
    for keywords,contexts,augmentations,image_names,image_paths in dataloader:
        #generate embeddings for context + augmentation
        text_emds = list()
        tokens = list()
        for i, j in zip(contexts, augmentations):
            context_augmented = i + " " + j
            # Tokenize the input text
            input_ids = torch.tensor([tokenizer.encode(context_augmented,max_length=77,truncation=True)])
            tokens.append(input_ids)

        image_emds = list()
        paths = [i.split("#") for i in image_paths]
        for t,ps in zip(tokens,paths):
            t = t.to(device)
            t_emds = model(t, None, setting="text").text_embeds
            images = open_images(ps)
            i_emds = list()
            for k in images:
                input_image = k['pixel_values'].to(device)
                i_emds.append(model(None, input_image, setting="image").image_embeds)

            i_emds = torch.stack(i_emds).squeeze().to(device)
            similarities = torch.nn.functional.pairwise_distance(t_emds, i_emds)
            similarities = similarities.cpu()
            similarities = similarities.detach().numpy()
            total+=1
            if int(np.argmin(similarities,axis=0))==0:
                correct+=1

    return correct/total


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
    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin

    def forward(self, image_embeddings, text_embeddings,device):
        # calculate positive distance between matching image and text embeddings
        positive_distance = (image_embeddings - text_embeddings).pow(2).sum(1).to(device)

        # calculate negative distance between all other image and text embeddings
        negative_distance = torch.zeros(image_embeddings.size(0)).to(device)

        for i in range(image_embeddings.size(0)):
            for j in range(image_embeddings.size(0)):
                if i != j:
                    negative_distance[i] += (image_embeddings[i] - text_embeddings[j]).pow(2).sum()
        negative_distance = negative_distance / (image_embeddings.size(0) - 1)
        # calculate loss
        loss = torch.mean((positive_distance - negative_distance + self.margin).clamp(min=0))
        return loss


def train_model(model,device,epoch,path_train,path_out):
    #train CLIP model for several epoches
    model.train()
    # Create the dataset
    with open(path_train, 'rb') as pickle_file:
        train_dataloader = pickle.load(pickle_file)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for i in range(epoch):
        print("--------------Training Epoch {}---------------".format(i))
        avg_loss = train_one_epoch(model, device,train_dataloader, optimizer)
        print("--------------Loss per instance{}---------------".format(avg_loss))
        filepath = path_out+"/inferencemodel"+str(i)
        torch.save(model.state_dict(), filepath)
        print("--------------Model saved at {}---------------".format(filepath))

        state = {'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}
        filepath = path_out + "/trainingmodel" + str(i)
        torch.save(state, filepath)
        print("--------------Model saved at {}---------------".format(filepath))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument('--epoch',help = "epoch")
    parser.add_argument('--train',help = 'path to train dataloader')
    parser.add_argument('--dev', help='path to dev dataloader')
    parser.add_argument('--test', help='path to test dataloader')
    parser.add_argument('--device',help="cuda to be used")
    parser.add_argument('--output',help = "path to save the model")
    parser.add_argument('--mode',help='train to train the model, test to evaluate the model')
    args = parser.parse_args()

    device_str = "cuda:" + str(args.device)
    device = torch.device(device_str)

    model = clip_model()
    model = model.to(device)

    if args.mode == 'train':
        train_model(model, device=device, epoch=5, path_train=args.train, path_out=args.output)
    elif args.mode == 'test':

        with open(args.dev, 'rb') as pickle_file:
            dev_dataloader = pickle.load(pickle_file)

        print("--------------Evaluation On Dev Using Original Model---------------")
        accuracy = evaluate(model, device, dev_dataloader)
        print("--------------Accuracy {}---------------".format(accuracy))

        for i in range(int(args.epoch)):
            filepath = args.output + "/inferencemodel" + str(i)
            model.load_state_dict(torch.load(filepath))
            print("--------------Evaluation On Dev---------------")
            accuracy = evaluate(model,device, dev_dataloader)
            print("--------------Accuracy {}---------------".format(accuracy))
    else:
        print("Wrong mode")