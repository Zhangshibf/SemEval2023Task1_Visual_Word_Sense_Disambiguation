#fine tune CLIP model
from dataloader_it import *
from PIL import Image
import argparse
from PIL import ImageFile
import torchvision.transforms as transforms
from torch import nn
from transformers import CLIPProcessor, CLIPVisionModelWithProjection,CLIPTokenizer, CLIPTextModelWithProjection
import torch
import PIL
from transformers import VisionTextDualEncoderModel
from transformers import AutoProcessor
from torch import optim

class clip_model(nn.Module):

    def __init__(self,device):
        super(clip_model, self).__init__()
        self.device = device
        self.model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        self.processor = AutoProcessor.from_pretrained("clip-italian/clip-italian")

    def forward(self, text, image,setting):
        if setting == "text":
            text_outputs = self.embed_texts(text)
            return text_outputs

        elif setting == "image":
            image_outputs = self.embed_images(image)
            return image_outputs

    def embed_texts(self,texts):
        inputs = self.processor(text=texts, padding="longest")
        input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(dim=0).to(self.device)
        attention_mask = torch.tensor(inputs["attention_mask"]).unsqueeze(dim=0).to(self.device)

        with torch.no_grad():
            embeddings = self.model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
        return embeddings

    def embed_images(self,images):
        inputs = self.processor(images=images)
        pixel_values = torch.tensor(np.array(inputs["pixel_values"])).to(self.device)

        with torch.no_grad():
            embeddings = self.model.get_image_features(pixel_values=pixel_values)
        return embeddings


def evaluate(model,device, dataloader,prediction_path):
    #use normalized dot product
    model.eval()
    for keywords,contexts,augmentations,image_names,image_paths in dataloader:
        image_names = [i.split("#") for i in image_names]
        texts = list()
        for k,c,a in zip(keywords,contexts, augmentations):
            #text = "Questa è una foto di " +k +" "+c
            texts.append(c)

        paths = [i.split("#") for i in image_paths]
        for keyword,context,t,ps in zip(keywords,contexts,texts,paths):
            t_emds = model(t, None, setting="text")
            images = open_images(ps)
            i_emds = model(None, images, setting="image")
            t_emds = t_emds / t_emds.norm(dim=1, keepdim=True)
            i_emds = i_emds / i_emds.norm(dim=1, keepdim=True)
            similarities = torch.matmul(t_emds, i_emds.transpose(0, 1))
            similarities = similarities.cpu()
            similarities = similarities.detach().numpy()

            #write output
            indices = np.argsort(similarities)[::-1]
            sorted = np.take(image_names, indices)
            sorted = np.flip(sorted)
            string = "\t".join(sorted.tolist()[0])+"\n"

            with open(prediction_path,"a") as file:
                file.write(string)
                file.close()

def open_images(image_paths):
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
        images.append(image)

    return images



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument('--dataset', help="dataset used to predict result")
    parser.add_argument('--device', help="cuda number")
    parser.add_argument("--output",help = "path to save the prediction")
    args = parser.parse_args()

    device_str = "cuda:" + args.device
    device = torch.device(device_str)

    model = clip_model(device = device)
    model = model.to(device)
    prediction_path = args.output
    PIL.Image.MAX_IMAGE_PIXELS = 933120000000000000000000000000000000000

    dataset_path = args.dataset
    with open(dataset_path, 'rb') as pickle_file:
        dataloader = pickle.load(pickle_file)
        pickle_file.close()

    print("--------------Evaluation---------------")
    evaluate(model,device, dataloader,prediction_path)
    print("--------------Evaluation Finished---------------")
