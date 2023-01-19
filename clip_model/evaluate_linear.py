from train_linear import *
from load_data import *
from PIL import Image
import argparse
from PIL import ImageFile
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPVisionModelWithProjection,CLIPTokenizer, CLIPTextModelWithProjection
import torch
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 93312000000000
def evaluate(model,nn, device, dataloader):
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
            t_emds = model(t, None, setting="text").text_embeds
            t_emds = nn(t_emds, image=None, setting="text")

            images = open_images(ps)
            i_emds = list()
            for k in images:
                input_image = k['pixel_values'].to(device)
                i_emd = model(None, input_image, setting="image").image_embeds
                i_emd = nn(None,i_emd,setting ="image")
                i_emds.append(i_emd)
            i_emds = torch.stack(i_emds).squeeze().to(device)

            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            similarities = cos(t_emds,i_emds)
            similarities = similarities.cpu()
            similarities = similarities.detach().numpy()
            total+=1
            rank = int(np.argsort(np.argsort(similarities))[0][0])
            print(similarities)
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
    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument('--test', help='path to test dataloader')
    parser.add_argument('--device',help="cuda to be used")
    parser.add_argument('--path',help = "path to the saved model")

    args = parser.parse_args()

    device_str = "cuda:" + str(args.device)
    device = torch.device(device_str)

    encoders = clip_model()
    encoders = encoders.to(device)
    nn = simple_nn()
    nn.load_state_dict(torch.load(args.path,map_location=device))
#    nn = nn.to(device)
    dataset_path = args.path
    with open("/home/CE/zhangshi/dataloader_8/dev.pk", 'rb') as pickle_file:
        dev_dataloader = pickle.load(pickle_file)
        pickle_file.close()

    evaluate(encoders,nn, device=device,dataloader=dev_dataloader)

