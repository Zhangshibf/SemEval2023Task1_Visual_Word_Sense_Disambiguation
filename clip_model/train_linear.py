from model import clip_model,simple_nn
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

def train_one_epoch(model,nn,device,dataloader,optimizer):
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
            text_emd = nn(outputs.text_embeds,image = None,setting = "text")
            text_emds.append(text_emd)

        image_emds = list()
        paths = [i.split("#") for i in image_paths]
        # each text corresponds to ten images. One image is positive sample and the rest nine are negative samples.
        paths = [item for sublist in paths for item in sublist]
        images = open_images(paths)
        for k in images:
            input_image = k['pixel_values']
            input_image = input_image.to(device)
            outputs = model(None, input_image, setting="image")
            image_emd = nn(text = None, image=outputs.image_embeds, setting="image")
            image_emds.append(image_emd)

        image_emds = torch.stack((image_emds)).squeeze(dim=1)
        text_emds = torch.stack((text_emds)).squeeze(dim=1)
        text_emds = torch.stack((text_emds,text_emds,text_emds,text_emds,text_emds,text_emds,text_emds,text_emds,text_emds,text_emds), dim=1).reshape(text_emds.size()[0]*10,text_emds.size()[1])
        print(text_emds)
        print(text_emds.size())
        image_emds = image_emds.to(device)
        text_emds = text_emds.to(device)
        labels = torch.tensor([1,-1,-1,-1,-1,-1,-1,-1,-1,-1], dtype=torch.float32).repeat(image_emds.size()[0])

        perm = torch.randperm(image_emds.size()[0])
        image_emds = image_emds[perm]
        text_emds = text_emds[perm]
        labels=labels[perm]
        cosine_loss = torch.nn.CosineEmbeddingLoss()
        loss_per_batch = cosine_loss(image_emds,text_emds,labels)

        loss+=float(loss_per_batch)
        model.zero_grad()
        loss_per_batch.backward()
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


def train_and_save_model(model,nn,device,epoch,path_train,path_out,optimizer):
    model.train()
    with open(path_train, 'rb') as pickle_file:
        train_dataloader = pickle.load(pickle_file)
    optimizer = optimizer

    for i in range(epoch):
        print("--------------Training Epoch {}---------------".format(i))
        avg_loss = train_one_epoch(model,nn, device,train_dataloader, optimizer)
        print("--------------Loss per instance{}---------------".format(avg_loss))
        filepath = path_out+"/inferencemodel"+str(i)
        torch.save(nn.state_dict(), filepath)
        print("--------------Model saved at {}---------------".format(filepath))

        state = {'epoch': epoch,
            'state_dict': nn.state_dict(),
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

    encoders = clip_model()
    encoders.text_encoder.requires_grad_(False)
    encoders.image_encoder.requires_grad_(False)
    encoders = encoders.to(device)
    nn = simple_nn()
    nn = nn.to(device)

    opt = optim.Adam(nn.parameters(), lr=0.0001, weight_decay=0.2)
    train_and_save_model(encoders,nn, device=device, epoch=int(args.epoch), path_train=args.train, path_out=args.output,optimizer=opt)

