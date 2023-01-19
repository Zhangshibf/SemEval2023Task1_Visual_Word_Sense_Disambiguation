from model import clip_model
from load_data import *
from PIL import Image
import argparse
from PIL import ImageFile
import torchvision.transforms as transforms
from torch import nn
from transformers import CLIPProcessor, CLIPVisionModelWithProjection,CLIPTokenizer, CLIPTextModelWithProjection
import torch
from torch import optim
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 93312000000000

def train_one_epoch(model,device,dataloader,optimizer,loss_mode):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",model_max_length=77)
    loss = 0
    for keywords,contexts,augmentations,image_names,image_paths in dataloader:
        text_emds = list()
        for i, j in zip(contexts, augmentations):
            context_augmented = i + " " + j
            # Tokenize the input text
            input_ids = torch.tensor([tokenizer.encode(context_augmented, max_length=77, truncation=True)])
            input_ids = input_ids.to(device)
            outputs = model(input_ids, None, setting="text")
            text_emds.append(outputs.text_embeds)

        if loss_mode == "pretraining":
            image_emds = list()
            #8 text embeddings
            #80 images
            #paths is a list of 8 positive images
            paths = [i.split("#")[0] for i in image_paths]
            #these are positive images
            images = open_images(paths)
            for k in images:
                input_image = k['pixel_values']
                input_image = input_image.to(device)
                outputs = model(None, input_image, setting="image")
                image_emds.append(outputs.image_embeds)

            image_emds = torch.stack((image_emds)).squeeze(dim=1)
            text_emds = torch.stack((text_emds)).squeeze(dim=1)
            #size = (B,E)
            image_emds = image_emds.to(device)
            text_emds = text_emds.to(device)
            loss_per_batch = pretraining_loss(image_emds,text_emds)
        elif loss_mode == "contrastive":
            image_emds = list()
            paths = [i.split("#") for i in image_paths]
            # each text corresponds to ten images. One image is positive sample and the rest nine are negative samples.
            paths = [item for sublist in paths for item in sublist]
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
            loss_per_batch = contrastive_loss(image_emds, text_emds)

        loss+=float(loss_per_batch)
        model.zero_grad()
        loss_per_batch.backward()
#        torch.nn.utils.clip_grad_norm_(model.parameters(),0.01)
        optimizer.step()

    return loss


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

def pretraining_loss(image_embeddings, text_embeddings):
    #(B,E)
    # Calculate the dot product between every image and every text embedding in the batch
    dot_products = torch.einsum('ab,cd->ac', [image_embeddings.div(image_embeddings.norm(dim=1, keepdim=True)),
                                              text_embeddings.div(text_embeddings.norm(dim=1, keepdim=True))])

    # Calculate the loss for each image in the batch
    image_losses = -torch.log(torch.exp(dot_products.diagonal()) / (torch.sum(torch.exp(dot_products), dim=1)))

    # Calculate the loss for each text in the batch
    text_losses = -torch.log(torch.exp(dot_products.diagonal()) / (torch.sum(torch.exp(dot_products), dim=0)))

    loss = torch.mean(image_losses) + torch.mean(text_losses)
    return loss

def contrastive_loss(image_embeddings, text_embeddings, margin=1.0):
    B, E = text_embeddings.size()
    text_embeddings = text_embeddings.view(B, 1, E)
    image_embeddings = image_embeddings.view(B, 10, E)
    positive_embeddings = image_embeddings[:, 0, :]
    negative_embeddings = image_embeddings[:, 1:, :]

    text_embeddings = text_embeddings / torch.norm(text_embeddings, dim=-1, keepdim=True)
    positive_embeddings = positive_embeddings / torch.norm(positive_embeddings, dim=-1, keepdim=True)
    negative_embeddings = negative_embeddings / torch.norm(negative_embeddings, dim=-1, keepdim=True)

    positive_similarity = torch.sum(text_embeddings * positive_embeddings, dim=-1)
    negative_similarity = torch.sum(text_embeddings * negative_embeddings, dim=-1)

    positive_loss = torch.clamp(margin - positive_similarity, min=0.0)
    negative_loss = torch.clamp(negative_similarity - margin, min=0.0)

    contrastive_loss = torch.mean(positive_loss + torch.sum(negative_loss, dim=-1))
    return contrastive_loss

def train_and_save_model(model,device,epoch,path_train,path_out,optimizer,loss):
    model.train()
    with open(path_train, 'rb') as pickle_file:
        train_dataloader = pickle.load(pickle_file)
    optimizer = optimizer

    for i in range(epoch):
        print("--------------Training Epoch {}---------------".format(i))
        avg_loss = train_one_epoch(model, device,train_dataloader, optimizer,loss)
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
    parser.add_argument('--loss',help='there are two types of loss. pretraining or contrastive')
    args = parser.parse_args()

    device_str = "cuda:" + str(args.device)
    device = torch.device(device_str)

    model = clip_model()
    model = model.to(device)
#    state = torch.load("/home/CE/zhangshi/SemEval23/clipgradient//trainingmodel0", map_location = device)
#    model.load_state_dict(state['state_dict'])

    opt = optim.Adam(model.parameters(), lr=5e-7, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    train_and_save_model(model, device=device, epoch=int(args.epoch), path_train=args.train, path_out=args.output,optimizer=opt,loss = args.loss)

