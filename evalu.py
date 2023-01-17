#fine tune CLIP model
from load_data import *
from PIL import Image
import argparse
from PIL import ImageFile
import torchvision.transforms as transforms
from torch import nn
from transformers import CLIPProcessor, CLIPVisionModelWithProjection,CLIPTokenizer, CLIPTextModelWithProjection
import torch
from torch import optim
import clip

class clip_model(nn.Module):
    def __init__(self):
        super(clip_model, self).__init__()
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.linear1 = nn.Linear(512,300)
        self.linear2 = nn.Linear(512,300)

    def forward(self, text, image,setting):
        if setting == "text":
            text_outputs = self.text_encoder(text).text_embeds
#            text_emds = self.linear1(text_outputs)
            return text_outputs

        elif setting == "image":
            image_outputs = self.image_encoder(image).image_embeds
#            image_emds = self.linear2(image_outputs.image_embeds)
            return image_outputs

def train_one_epoch(model,device,dataloader,optimizer):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",model_max_length=77)
    loss = 0
#    criterion = ContrastiveLoss()
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
            text_emds.append(outputs)

        image_emds = list()
        paths = [i.split("#")[0] for i in image_paths]
        #these are positive images
        images = open_images(paths)
        for k in images:
            input_image = k['pixel_values']
            input_image = input_image.to(device)
            outputs = model(None, input_image, setting="image")
            image_emds.append(outputs)

        image_emds = torch.stack((image_emds)).squeeze(dim=1)
        text_emds = torch.stack((text_emds)).squeeze(dim=1)
        image_emds = image_emds.to(device)
        text_emds = text_emds.to(device)

        loss_per_batch = pretraining_loss(image_emds,text_emds)
        loss+=float(loss_per_batch)

        model.zero_grad()
        loss_per_batch.backward()
        optimizer.step()

    return loss


def evaluate(model,device, dataloader):
    #use normalized dot product
    model.eval()
    correct = 0
    total = 0
    mrr = 0
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",model_max_length=77)
    for keywords,contexts,augmentations,image_names,image_paths in dataloader:
        tokens = list()
        for i, j in zip(contexts, augmentations):
            context_augmented = i + " " + j
            # Tokenize the input text
            input_ids = torch.tensor([tokenizer.encode(context_augmented,max_length=77,truncation=True)])
            tokens.append(input_ids)

        paths = [i.split("#") for i in image_paths]
        for t,ps in zip(tokens,paths):
            t = t.to(device)
            t_emds = model(t, None, setting="text")
            images = open_images(ps)
            i_emds = list()
            for k in images:
                input_image = k['pixel_values'].to(device)
                i_emds.append(model(None, input_image, setting="image"))

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
                print("c")
            else:
                print("no")
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



if __name__ == "__main__":
    device_str = "cuda:" + str(2)
    device = torch.device(device_str)

    model = clip_model()
    model.text_encoder.requires_grad_(False)
    model.image_encoder.requires_grad_(False)
    model = model.to(device)

    with open("/home/CE/zhangshi/dataloader_8/dev.pk", 'rb') as pickle_file:
        dev_dataloader = pickle.load(pickle_file)

#    filepath = "/home/CE/zhangshi/SemEval23/contrastive/inferencemodel3"
#    filepath = "/home/CE/zhangshi/SemEval23/clipgradient//inferencemodel1"
#    filepath = "/home/CE/zhangshi/SemEval23/clipgradient//inferencemodel3"
#    filepath = "/home/CE/zhangshi/SemEval23/clipgradient/inferencemodel12"
    filepath = "/home/CE/zhangshi/SemEval23/clip_model/new_loss/inferencemodel0"
#    model.load_state_dict(torch.load(filepath))
    print("--------------Evaluation---------------")
    hit_rate,mrr = evaluate(model,device, dev_dataloader)
    print("--------------Accuracy {}---------------".format(hit_rate))
    print("--------------MRR {}---------------".format(mrr))
