import os
import json
import math
import torch
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertTokenizerFast
from transformers import ViltFeatureExtractor, ViltConfig
from model import CustomModel
from data import ImageTextDataset, custom_collate
from torchmetrics.functional import retrieval_hit_rate, retrieval_reciprocal_rank


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument('--config', help="Configuration file in JSON format", default="config.json")
    parser.add_argument('--test_dir', help="Input the directory of test images and file in txt format", default="test.data.txt")
    args = parser.parse_args()

    # Loading Configuration File (dictionary)
    config = json.load(open(args.config))
    print("Configuration", config)
    checkpoint = config["model"]
    if config.test_dir:
        data_dir = config.test_dir
    else:
        data_dir = config["data_dir"]
        
    percentage = config["data_size"]/100
    lr = config["lr"]
    epochs = config["epochs"]
    if config.test_dir:
        data_df = pd.read_csv(os.path.join(data_dir, "test.data.txt"), sep='\t', header=None, names=['word', 'description', 'image_0', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'image_8', 'image_9'])
        label_df = pd.read_csv(os.path.join(data_dir, "test.gold.txt"), sep='\t', header=None, names=['gold_image'])
    
    else:
        data_df = pd.read_csv(os.path.join(data_dir, "train.data.txt"), sep='\t', header=None, names=['word', 'description', 'image_0', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'image_8', 'image_9'])
        label_df = pd.read_csv(os.path.join(data_dir, "train.gold.txt"), sep='\t', header=None, names=['gold_image'])
        df = data_df.join(label_df)

    
    data_df['images'] = data_df.iloc[:,2:].values.tolist()
    data_df = data_df.drop(columns= ['image_0', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'image_8', 'image_9'])
    data_size = math.ceil(percentage * len(df))
    print("="*100)
    print(f"\nTaking {percentage * 100} percentage of Total Dataset\n")
    df = df[0:data_size]
    print(f"Total Length of the Dataset {len(df)}\n")

    train, _ = train_test_split(df, test_size=0.20, random_state= 42)
    test, valid = train_test_split(_, test_size=0.50, random_state= 42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    feature_extractor = ViltFeatureExtractor.from_pretrained(checkpoint)
    processor = {
        'tokenizer': tokenizer,
        'feature_extractor': feature_extractor
    }


    model = CustomModel()
    model=CustomModel(config = ViltConfig.from_pretrained(f"resources/{config['data_size']}/{epochs}/{lr}/", output_attentions=True,output_hidden_states=True, num_images=10, num_labels=10,  problem_type="multi_label_classification"))


    # Create the dataset
    test_ds = ImageTextDataset(os.path.join(data_dir, "test_images_v1"), test, data_type="test", device = device, text_augmentation=False)
    # Create the dataloader
    test_dataloader = DataLoader(test, shuffle=True, batch_size=4, collate_fn=lambda batch: custom_collate(batch, processor))
    
    num_test_steps = epochs * len(test_dataloader)

    progress_bar_test = tqdm(range(num_test_steps))
    

    model.eval()
    total_loss = 0
    print(len(test_dataloader))
    
    #Initialize Hit Rate and Reciprocal Rank
    hit_rate = 0
    r_rank = 0

    submission = []
    # print(list(test['images']))
    for i, batch in enumerate(test_dataloader):
    # print(batch)
    # batch = {k: v.to(device) for k, v in batch.items()}
        batch.to(device)
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], pixel_values=batch['pixel_values'], labels=batch['labels'])

        logits = outputs.logits.view(-1, 10)
        values, indices = logits[0].topk(10)
        
        res = [list(test['images'])[i][index.item()] for value, index in zip(values, indices)]
        submission.append('\t'.join(res))
        # print(logits, batch['labels'])

        # print(retrieval_hit_rate(logits,batch['labels'], k=1))
        # print(retrieval_reciprocal_rank(logits,batch['labels']))

        hit_rate += retrieval_hit_rate(logits,batch['labels'], k=1)
        r_rank += retrieval_reciprocal_rank(logits,batch['labels'])



    print(hit_rate/len(test_dataloader))
    print(r_rank/len(test_dataloader))

    with open(r'submission.txt', 'w') as f:
        f.write('\n'.join(submission))