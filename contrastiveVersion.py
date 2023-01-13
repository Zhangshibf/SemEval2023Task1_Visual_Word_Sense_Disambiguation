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

class clip_model(nn.Module):
    def __init__(self):
        super(clip_model, self).__init__()
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, text, image,setting):
        setting_types = ["text","image"]
        if setting not in setting_types:
            raise ValueError("Invalid data type. Expected one of: %s" % setting_types)

        if setting == "text":
            text_outputs = self.text_encoder(text)
            return text_outputs

        elif setting == "image":
            image_outputs = self.image_encoder(image)
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
            text_emds.append(outputs.text_embeds)

        image_emds = list()
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
        image_emds = image_emds.to(device)
        text_emds = text_emds.to(device)

        loss_per_batch = pretraining_loss(image_emds,text_emds)
        loss+=float(loss_per_batch)
        model.zero_grad()

        # Backpropagate the loss and update the model weights
        loss_per_batch.backward()
        optimizer.step()

    return loss


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

#change this into 1 positive vs 9 negative
def pretraining_loss(image_embeddings, text_embeddings):
    epsilon = 1e-8
    # Calculate the dot product between every image and every text embedding in the batch
    dot_products = torch.einsum('ab,cd->ac', [image_embeddings.div(image_embeddings.norm(dim=1, keepdim=True)),
                                              text_embeddings.div(text_embeddings.norm(dim=1, keepdim=True))])

    # Calculate the loss for each image in the batch
    image_losses = -torch.log(torch.exp(dot_products.diagonal()) / (torch.sum(torch.exp(dot_products), dim=1) + epsilon))

    # Calculate the loss for each text in the batch
    text_losses = -torch.log(torch.exp(dot_products.diagonal()) / (torch.sum(torch.exp(dot_products), dim=0) + epsilon))

    return torch.mean(image_losses) + torch.mean(text_losses)


def train_model(model,device,epoch,path_train,path_out,optimizer):
    model.train()
    with open(path_train, 'rb') as pickle_file:
        train_dataloader = pickle.load(pickle_file)
    optimizer = optimizer

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
        opt = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
        train_model(model, device=device, epoch=int(args.epoch), path_train=args.train, path_out=args.output,optimizer=opt)

    elif args.mode == 'test':

        with open(args.dev, 'rb') as pickle_file:
            dev_dataloader = pickle.load(pickle_file)

        print("--------------Evaluation On Dev Using Original Model---------------")
        hit_rate,mrr = evaluate(model, device, dev_dataloader)
        print("--------------Accuracy {}---------------".format(hit_rate))
        print("--------------MRR {}---------------".format(mrr))

        for i in range(int(args.epoch)):
            filepath = args.output + "/inferencemodel" + str(i)
            model.load_state_dict(torch.load(filepath))
            print("--------------Evaluation On Dev---------------")
            hit_rate,mrr = evaluate(model,device, dev_dataloader)
            print("--------------Accuracy {}---------------".format(hit_rate))
            print("--------------MRR {}---------------".format(mrr))
    else:
        print("Wrong mode")
