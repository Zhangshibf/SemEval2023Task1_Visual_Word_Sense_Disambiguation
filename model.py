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
        self.linear1 = nn.Linear(1024,300)
        self.linear2 = nn.Linear(300,2)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, text, image):
        text_outputs = self.text_encoder(text).text_embeds
        image_outputs = self.image_encoder(image).image_embeds
        concat = torch.cat((text_outputs,image_outputs), 1)
        out1 = self.linear1(concat)
        out2 = self.linear2(out1)
        prediction = self.softmax(out2)

        return prediction

def train_one_epoch(model,device,dataloader,optimizer):
    loss_total = 0
    correct_total = 0
    instance_total =0
    # Train CLIP model for one epoch
    for keywords,contexts,augmentations,image_names,image_paths in dataloader:
        #this is one batch
        tokens = list()
        texts = list()
        images = list()
        labels = list()
        p_label = torch.tensor([1,0])
        n_label = torch.tensor([0,1])
        for i, j,k in zip(contexts, augmentations,image_paths):
            paths = k.split("#")
            img = open_images(paths)
            for k in range(len(img)):
                input_image = img[k]['pixel_values']
                input_image = input_image
                images.append(input_image)
                if k == 0:
                    labels.append(p_label)
                else:
                    labels.append(n_label)

            context_augmented = i + " " + j
            texts.append(context_augmented)
        instance_total+=len(texts)
        texts = [[i]*10 for i in texts]
        texts = [item for sublist in texts for item in sublist]
        tokens= clip.tokenize(texts,truncate = True).to(device)
        tokens = tokens.to(device)

        images = torch.stack(images).squeeze().to(device)
        labels = torch.stack(labels).squeeze().to(device)
        prediction = model(tokens,images)
        correct_per_batch = calculate_correct(prediction,labels)
        correct_total+=correct_per_batch
        loss_per_batch = torch.nn.functional.binary_cross_entropy(prediction.float(),labels.float())
        loss_total+=float(loss_per_batch)

        model.zero_grad()
        loss_per_batch.backward()
        optimizer.step()

    return correct_total/instance_total,loss_per_batch

def calculate_correct(prediction,labels):
    correct = 0
    pre = prediction.squeeze().tolist()
    num = len(pre)/10
    #ten images, the first one is the correct one.
    for i in range(int(num)):
        scores = list()
        for j in range(i,i+10):
            scores.append(pre[j][0])
        if np.argmax(np.array(scores))==0:
            correct+=1
    print(correct)

    return correct

def calculate_mrr(prediction,labels):
    prediction = prediction.tolist()
    labels = labels.tolist()
    mrr = 0
    #1 text, 10 images
    #prediction[:10],prediction[10:20],...
    num = len(prediction)/10
    for i in range(num):
        pred_per_insteance = prediction[int(i*10),int((i+1)*10)]
        label_per_insteance = labels[int(i*10),int((i+1)*10)]
        positive = np.array([i[0] for i in pred_per_insteance])
        rank = int(np.argsort(np.argsort(positive))[0][0])
        mrr+=1/(10-rank)

    return mrr


def evaluate(model,device, dataloader):
    #now use normalized dot product instead of cosine similarity
    #cosine similarity instead of L2 distance
    model.eval()
    mrr_total = 0
    correct_total = 0
    instance_total = 0
    b = 0
    # Train CLIP model for one epoch
    for keywords, contexts, augmentations, image_names, image_paths in dataloader:
        instance_total+=len(keywords)
        # this is one batch
        texts = list()
        images = list()
        labels = list()
        p_label = torch.tensor([1, 0])
        n_label = torch.tensor([0, 1])
        for i, j, k in zip(contexts, augmentations, image_paths):
            paths = k.split("#")
            img = open_images(paths)
            for k in range(len(img)):
                input_image = img[k]['pixel_values']
                input_image = input_image
                images.append(input_image)
                if k == 0:
                    labels.append(p_label)
                else:
                    labels.append(n_label)

            context_augmented = i + " " + j
            texts.append(context_augmented)
        instance_total += len(texts)
        texts = [[i] * 10 for i in texts]
        texts = [item for sublist in texts for item in sublist]
        tokens = clip.tokenize(texts,truncate = True).to(device)
        tokens = tokens.to(device)

        images = torch.stack(images).squeeze().to(device)
        labels = torch.stack(labels).squeeze().to(device)
        prediction = model(tokens, images)
        correct_per_batch = calculate_correct(prediction, labels)
        mrr_per_batch = calculate_mrr(prediction,labels)

        correct_total += correct_per_batch
        mrr_total += mrr_per_batch

    hit_rate = correct_total/instance_total
    mrr = mrr_total/instance_total

    return hit_rate,mrr

def open_images(image_paths):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    transform = transforms.Compose(
        [transforms.Resize([720, 900]), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

def train_model(model,device,epoch,path_train,path_out):
    #train CLIP model for several epoches
    model.train()
    # Create the dataset
    with open(path_train, 'rb') as pickle_file:
        train_dataloader = pickle.load(pickle_file)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for i in range(epoch):
        print("--------------Training Epoch {}---------------".format(i))
        avg_accuracy,avg_loss = train_one_epoch(model, device,train_dataloader, optimizer)
        print("--------------Loss {}---------------".format(avg_loss))
        print("--------------Accuracy {}---------------".format(avg_accuracy))
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