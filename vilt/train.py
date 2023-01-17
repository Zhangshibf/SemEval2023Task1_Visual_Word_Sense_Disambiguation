import os
import math
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from transformers import ViltFeatureExtractor, ViltConfig
from transformers import AdamW,get_scheduler
from tqdm.auto import tqdm
from model import CustomModel
from data import ImageTextDataset, custom_collate


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument('--config', help="Configuration file in JSON format", default="config.json")
    args = parser.parse_args()

    # Loading Configuration File (dictionary)
    config = json.load(open(args.config))
    print("Configuration", config)
    data_dir = config["data_dir"]
    percentage = config["data_size"]/100
    lr = config["lr"]
    epochs = config["epochs"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = "dandelin/vilt-b32-mlm"
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    feature_extractor = ViltFeatureExtractor.from_pretrained(checkpoint)
    processor = {
        'tokenizer': tokenizer,
        'feature_extractor': feature_extractor
    }
    data_df = pd.read_csv(os.path.join(data_dir, "trial.data.txt"), sep='\t', header=None, names=['word', 'description', 'image_0', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'image_8', 'image_9'])
    label_df = pd.read_csv(os.path.join(data_dir, "trial.gold.txt"), sep='\t', header=None, names=['gold_image'])
    
    data_df['images'] = data_df.iloc[:,2:].values.tolist()
    data_df = data_df.drop(columns= ['image_0', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'image_8', 'image_9'])
    df = data_df.join(label_df)
    data_size = math.ceil(percentage * len(df))
    print("="*100)
    print(f"\nTaking {percentage * 100} percentage of Total Dataset\n")
    df = df[0:data_size]
    print(f"Total Length of the Dataset {len(df)}\n")

    train, _ = train_test_split(df, test_size=0.20, random_state= 42)
    test, valid = train_test_split(_, test_size=0.50, random_state= 42)
    print(f"Train Set {len(train)}, Validation Set {len(valid)}, Test Set {len(test)}\n")
    print(f"Keyword: '{train['word'][0]}'\n")
    print(f"Description: '{train['description'][0]}'\n")
    print(f"List of 10 Images: {train['images'][0]}\n")
    
    #loding model
    print("===================Loading Model=======================")
    model=CustomModel(config = ViltConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True, num_images=10, num_labels=10,  problem_type="multi_label_classification"))
    optimizer = AdamW(model.parameters(), lr=lr)
    model.to(device)
    print(model.config)

    # Create the dataset
    train_ds = ImageTextDataset(os.path.join(data_dir, "trial_images_v1"), train, data_type="train",device = device, text_augmentation=True)
    valid_ds = ImageTextDataset(os.path.join(data_dir, "trial_images_v1"), valid, data_type="valid",device = device, text_augmentation=True)
    # Create the dataloader
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=2, collate_fn=lambda batch: custom_collate(batch, processor))
    valid_dataloader = DataLoader(valid_ds, shuffle=True, batch_size=2, collate_fn=lambda batch: custom_collate(batch, processor))
    
    print(f"Length of Train Dataloader {len(train_dataloader)}")
    print(f"Length of Valid Dataloader {len(valid_dataloader)}")
    
    # model.to(device)

    num_training_steps = epochs * len(train_dataloader)
    num_validing_steps = epochs * len(train_dataloader)

    progress_bar_train = tqdm(range(num_training_steps))
    progress_bar_valid = tqdm(range(num_validing_steps))

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )


    print(num_training_steps)
    min_loss = np.inf
    model.train()
    for i in range(epochs):
      
      total_loss = 0
      print(f"Epoch {i+1}")
      
      for batch in train_dataloader:
        batch.to(device)
        
        outputs = model(input_ids=batch['input_ids'], pixel_values=batch['pixel_values'], labels=batch['labels'])
        
        loss = outputs.loss
        
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar_train.update(1)

      model.eval()
      for batch in valid_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], pixel_values=batch['pixel_values'], labels=batch['labels'])

        total_loss += outputs.loss
        
        progress_bar_valid.update(1)

      total_loss /= len(valid_dataloader)
      print("Validation loss", total_loss)
      if total_loss < min_loss:
        print(min_loss, total_loss)
        min_loss = total_loss
        print(f"Saving Model After Epoch {i+1}")
        model.save_pretrained(f"resources/{config['data_size']}/{epochs}/{lr}/")
