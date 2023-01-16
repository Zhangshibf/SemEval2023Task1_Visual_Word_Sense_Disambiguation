from clip_model.model import clip_model
from load_data import *
from PIL import Image
import argparse
from PIL import ImageFile
import torchvision.transforms as transforms
from torch import nn
from transformers import CLIPProcessor, CLIPVisionModelWithProjection,CLIPTokenizer, CLIPTextModelWithProjection
import torch
from torch import optim

def evaluate(model,device, dataloader):
    #now use normalized dot product instead of cosine similarity
    #cosine similarity instead of L2 distance
    model.eval()
    correct = 0
    total = 0
    mrr = 0
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
            t_emds = t_emds / t_emds.norm(dim=1, keepdim=True)
            i_emds = i_emds / i_emds.norm(dim=1, keepdim=True)
            similarities = torch.matmul(t_emds, i_emds.transpose(0, 1))
            similarities = similarities.cpu()
            similarities = similarities.detach().numpy()
            total+=1

            rank = int(np.argsort(np.argsort(similarities))[0][0])
            if int(rank) == 9:
                correct+=1
            mrr+=1/(10-rank)
    hit_rate = correct/total
    mrr = mrr/total

    return hit_rate,mrr

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
