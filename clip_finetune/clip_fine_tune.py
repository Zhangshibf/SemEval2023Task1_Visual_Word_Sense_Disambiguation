import os
import argparse
import json
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = 1000000000
import os
import clip
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import AdamW,get_scheduler
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW,get_scheduler
from tqdm.auto import tqdm
from data import ImageTextDataset



if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument('--epochs', type = int, help="Number of Epochs", default=5)
    parser.add_argument('--lr', type = float, help="Learning rate", default=5e-5)
    parser.add_argument('--no_augmentation', help="Augmenting Description Default (True)", action='store_false', default=True)
    parser.add_argument('--text_file', help="Input Train file in txt format", default="semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt")
    parser.add_argument('--gold_file', help="Input Gold file in txt format", default="semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt")
    parser.add_argument('--image_dir', help="Input the directory of train images", default="semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1")
    args = parser.parse_args()
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'DEVICE USED: {device}')
    loss_fct = nn.CrossEntropyLoss()
    # Prepare the inputs
    data_df = pd.read_csv(args.text_file, sep='\t', header=None, names=['word', 'description', 'image_0', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'image_8', 'image_9'])
    label_df = pd.read_csv(args.gold_file, sep='\t', header=None, names=['gold_image'])

    data_df['images'] = data_df.iloc[:,2:].values.tolist()
    data_df = data_df.drop(columns= ['image_0', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'image_8', 'image_9'])
    df = data_df.join(label_df)
    
    train, _ = train_test_split(df, test_size=0.2, random_state= 42)
    test, valid = train_test_split(_, test_size=0.50, random_state= 42)
    print(len(train), len(valid))

    # Create the dataset
    train_ds = ImageTextDataset(args.image_dir, train, data_type="train",device = device, text_augmentation=args.no_augmentation)
    valid_ds = ImageTextDataset(args.image_dir, valid, data_type="valid",device = device, text_augmentation=args.no_augmentation)
    # Create the dataloader
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=32)
    valid_dataloader = DataLoader(valid_ds, shuffle=True, batch_size=32)

    print(len(train_dataloader), len(valid_dataloader))
    model, preprocess = clip.load('ViT-B/32', device=device, jit = False)
    model.to(device)
    lr=args.lr
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    num_epochs = args.epochs

    path = f"checkpoints/{len(train)}/{num_epochs}/{lr}"
    if not os.path.exists(path):
        os.makedirs(path)

    num_training_steps = num_epochs * len(train_dataloader)
    num_valid_steps = num_epochs * len(valid_dataloader)

    progress_bar_train = tqdm(range(num_training_steps))
    progress_bar_valid = tqdm(range(num_valid_steps))

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    print(num_training_steps)

    total_loss = torch.tensor(float('inf')).to(device)

    train_losses = []
    eval_losses = []

    for i in range(num_epochs):
    
        print(f"epoch: {i+1}")
        print("=====================Training====================\n")
        train_loss = 0
        for idx, (context, images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            context = context.to(device)
            labels = labels.to(device)
            image_features_ = model.encode_image(images.flatten(0, 1)).reshape(*images.shape[:2], -1)

            text_features_ = model.encode_text(context.squeeze(1))
            image_features = image_features_ / image_features_.norm(dim=-1, keepdim=True)
            text_features = text_features_ / text_features_.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.unsqueeze(2)).squeeze() * model.logit_scale.exp()

            loss = loss_fct(similarity, torch.as_tensor(labels))
            train_loss += loss
            # print(loss)
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar_train.update(1)

        train_losses.append((train_loss/len(train_dataloader)).item())

        
        print("\n=====================Validation====================")
        eval_loss = 0
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

            eval_loss += loss
            progress_bar_valid.update(1)
        
        eval_losses.append((eval_loss/len(valid_dataloader)).item())

        

        if total_loss > eval_loss:
            total_loss = eval_loss

            
            print(f"Saving Model After Epoch {i+1} in {path}")
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':total_loss,
            }, f"{path}/best_model.pt")

    
    losses = {
    'train': train_losses,
    'eval': eval_losses
    }

    # Serializing json
    json_object = json.dumps(losses, indent=4)
    with open(f"{path}/loss.json", "w") as f:
        f.write(json_object)




