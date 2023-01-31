#fine tune CLIP model
from load_data_final_prediction import *
from PIL import Image
import argparse
from PIL import ImageFile
import torchvision.transforms as transforms
from torch import nn
from transformers import CLIPProcessor, CLIPVisionModelWithProjection,CLIPTokenizer, CLIPTextModelWithProjection
import torch
from torch import optim
import clip
def evaluate(model,preprocess,device, dataloader,prediction_path):
    #use normalized dot product
    model.eval()
    for keywords,contexts,augmentations,image_names,image_paths in dataloader:
        image_names = [i.split("#") for i in image_names]
        tokens = list()
        for k,c,a in zip(keywords,contexts, augmentations):

            try:
                context_augmented = c + " " + a
                #context_augmented = "This is a photo of" + k + ". " + c + ": " + a
            except:
                if c == "nan bread":
                    context_augmented = "Naan bread is a type of bread made with flour. It is a flatbread that is baked in a tandoor. Naan bread often looks like a tear drop. It is often covered in herbs and spices such as garlic to change the taste.Naan bread is made from basic bread ingredients like wheat flour, a leavening agent, salt, and butter or ghee."
                elif c == "nan river":
                    context_augmented = "The Nan River is a river in Thailand. It is one of the most important tributaries of the Chao Phraya River."

            input_ids = clip.tokenize(context_augmented, context_length=77, truncate=True)
            tokens.append(input_ids)

        paths = [i.split("#") for i in image_paths]
        for keyword,context,t,ps in zip(keywords,contexts,tokens,paths):
            t = t.to(device)
            t_emds = model.encode_text(t)
            image_inputs = torch.cat([preprocess(Image.open(img)).unsqueeze(0) for img in ps]).to(device)
            i_emds = model.encode_image(image_inputs)
#            i_emds = list()
#            for k in images:
#                input_image = k.to(device)
#                print(input_image.size())
#                i_emds.append(model.encode_image(input_image))

#            i_emds = torch.stack(i_emds).squeeze().to(device)
            t_emds = t_emds / t_emds.norm(dim=1, keepdim=True)
            i_emds = i_emds / i_emds.norm(dim=1, keepdim=True)
            similarities = torch.matmul(t_emds, i_emds.transpose(0, 1))
            similarities = similarities.cpu()
            similarities = similarities.detach().numpy()
            # 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

            #write output
            indices = np.argsort(similarities)[::-1]
            sorted = np.take(image_names, indices)
            sorted = np.flip(sorted)
            string = "\t".join(sorted.tolist()[0])+"\n"

            with open(prediction_path,"a") as file:
                file.write(string)
                file.close()

def open_images(preprocess,image_paths):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    transform = transforms.Compose(
        [transforms.Resize([1440, 1810]), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    images = list()
    for path in image_paths:
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert('RGB')
        image = preprocess(image)
        images.append(image)

    return images



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument('--dataset', help="dataset used to predict result")
    #/home/CE/zhangshi/SemEval23/clip_zeroshot/testset_dataloader/dataset.pk
    parser.add_argument('--device', help="cuda number")
    parser.add_argument("--output",help = "path to save the prediction")
    args = parser.parse_args()

    device_str = "cuda:" + args.device
    device = torch.device(device_str)

    model, preprocess = clip.load('ViT-B/32', device)
    model = model.to(device)
    prediction_path = args.output

    dataset_path = args.dataset
    #batch size of the data has to be one!!!
    with open(dataset_path, 'rb') as pickle_file:
        dataloader = pickle.load(pickle_file)
        pickle_file.close()

    print("--------------Evaluation---------------")
    evaluate(model,preprocess, device,dataloader,prediction_path)
    print("--------------Evaluation Finished---------------")
