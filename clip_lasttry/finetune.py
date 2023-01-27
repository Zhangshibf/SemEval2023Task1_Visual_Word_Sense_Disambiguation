import PIL
from PIL import Image
import clip
from transformers import AdamW,get_scheduler
from tqdm.auto import tqdm
from load_data import *
from PIL import Image
import argparse
from PIL import ImageFile
import torchvision.transforms as transforms
from torch import nn
from transformers import CLIPProcessor, CLIPVisionModelWithProjection,CLIPTokenizer, CLIPTextModelWithProjection
import torch
import PIL.Image


def train(device,train_dataloader,model_name = 'ViT-B/32',lr = 2e-5,num_epochs = 20):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    PIL.Image.MAX_IMAGE_PIXELS = 93312000000000

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", model_max_length=77)
    loss_fct = nn.CrossEntropyLoss()
    torch.autograd.set_detect_anomaly(True)
    model, preprocess = clip.load(model_name, device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    num_training_steps = num_epochs * len(train_dataloader)

    progress_bar_train = tqdm(range(num_training_steps))

    lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
    )

    total_loss = torch.tensor(float('inf')).to(device)

    for i in range(num_epochs):
        print(f"---------------epoch: {i + 1}---------------------")
    for idx, (keywords, contexts, augmentations, image_names, image_paths) in enumerate(train_dataloader):
        text_emds = list()
        for i, j in zip(contexts, augmentations):
            context_augmented = i + " " + j
            # Tokenize the input text
            input_ids = torch.tensor([tokenizer.encode(context_augmented, max_length=77, truncation=True)])
            input_ids = input_ids.to(device)
            outputs = model.encode_text(input_ids)
            text_emds.append(outputs)

        image_emds = list()
        paths = [i.split("#") for i in image_paths]
        # each text corresponds to ten images. One image is positive sample and the rest nine are negative samples.
        paths = [item for sublist in paths for item in sublist]
        images = open_images(paths)
        for k in images:
            input_image = k['pixel_values']
            input_image = input_image.to(device)
            outputs = model.encode_image(input_image)
            image_emds.append(outputs)

        image_emds = torch.stack((image_emds)).squeeze(dim=1)
        text_emds = torch.stack((text_emds)).squeeze(dim=1)
        image_features_ = image_emds.to(device)
        text_features_ = text_emds.to(device)
        image_features = image_features_ / image_features_.norm(dim=-1, keepdim=True)
        text_features = text_features_ / text_features_.norm(dim=-1, keepdim=True)

        labels = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32).repeat(
            image_emds.size()[0]).to(device)

        similarity = (image_features @ text_features.unsqueeze(2)).squeeze() * model.logit_scale.exp()
        loss = loss_fct(similarity, torch.as_tensor(labels))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar_train.update(1)


#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
def open_images(image_paths):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    transform = transforms.Compose(
        [transforms.Resize([770, 905]), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
    parser.add_argument('--epoch',help = "epoch")
    parser.add_argument('--train',help = 'path to train dataloader')
    parser.add_argument('--device',help="cuda to be used")
    parser.add_argument('--output',help = "path to save the model")
    args = parser.parse_args()

    torch.cuda.empty_cache()
    device_str = "cuda:" + str(args.device)
    device = torch.device(device_str)

    with open(args.train, 'rb') as pickle_file:
        train_dataloader = pickle.load(pickle_file)

    train(model_name='ViT-B/32', lr=2e-5, num_epochs=int(args.epoch), device=device, train_dataloader=train_dataloader)

    """
    
        batch_loss = 0
    for idx, (context, images, labels) in enumerate(valid_dataloader):
        images = images.to(device)
        context = context.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            image_features = model.encode_image(images.flatten(0, 1)).view(*images.shape[:2], -1)
            text_features = model.encode_text(context.squeeze(1))

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.unsqueeze(2)).squeeze() * model.logit_scale.exp()

        loss = loss_fct(similarity, torch.as_tensor(labels))

        batch_loss += loss
        progress_bar_valid.update(1)

    batch_loss = batch_loss / len(valid_dataloader)
    print(batch_loss)

    if total_loss > batch_loss:
        total_loss = batch_loss
        path = f"checkpoints/{len(train)}/{num_epochs}/{lr}"
        if not os.path.exists(path):
            os.makedirs(path)

        print(f"Saving Model After Epoch {i + 1}")
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{path}/best_model.pth")"""
