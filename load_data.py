import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import argparse
import torch
from nltk.corpus import wordnet as wn
import nltk
from sentence_transformers import SentenceTransformer, util
import numpy as np
class ImageTextDataset(Dataset):
    def __init__(self, data_dir, data_type,device,text_augmentation=False):
        self.device = device
        types = ["inaturalist", "train"]
        if data_type not in types:
            raise ValueError("Invalid data type. Expected one of: %s" % data_type)
        augmentation_types = [True,False]
        if text_augmentation not in augmentation_types:
            raise ValueError("Invalid augmentation type. Expected one of: %s" % augmentation_types)

        self.data_dir = data_dir
        self.image_path = list()
        self.image_name = list()

        if data_type == "inaturalist":
            # I will write this part later
            pass
        elif data_type == "train":
            # this is for the original train set of the task
            # reshape all images to size [1440,1810]
            self.transform = transforms.Compose([transforms.Resize([1440,1810]),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                 ])
            train_data = pd.read_csv(os.path.join(data_dir, "train.data.v1.txt"), sep="\t", header=None)
            label_data = pd.read_csv(os.path.join(data_dir, "train.gold.v1.txt"), sep="\t", header=None)
            keywords = list(train_data[0])
            contexts = list(train_data[1])

            self.keywords = keywords
            self.context = contexts
            image_filenames = list(label_data[0])
            for filename in image_filenames:
                self.image_name.append(filename)
                self.image_path.append(os.path.join(data_dir, "train_images_v1", filename))

        #text augmentation
        #an augmented text is composed of lemmas + definition from wordnet
        if text_augmentation:
            nltk.download('omw-1.4')
            nltk.download('wordnet')
            self.augmentation = list()
            sent_encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(self.device)
            for keyword,phrase in zip(self.keywords,self.context):

                #retrieve all possible augmented texts
                synsets = wn.synsets(keyword)
                augmented_texts = list()
                if len(synsets)!=0:
                    for synset in synsets:
                        augmented_text = ''
                        for lemma in synset.lemmas():
                            augmented_text += str(lemma.name()).replace('_', ' ') + ', '
                        augmented_text += synset.definition()
                        augmented_texts.append(augmented_text)
                    print(augmented_texts)

                if len(augmented_texts)>1:
                    #check which of the augmented texts is more similar to the short phrase
                    context_emb = sent_encoder.encode(phrase)
                    aug_emb = sent_encoder.encode(augmented_texts)
                    print(len(context_emb))
                    print(len(aug_emb))
                    scores = util.dot_score(context_emb, aug_emb)[0].tolist()
                    idx = np.argmax(scores)
                    self.augmentation.append(augmented_texts[idx])
                elif len(augmented_texts) = 1:
                    self.augmentation.append(augmented_texts[0])
                elif len(augmented_texts) = 0:
                    self.augmentation.append(phrase)
    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        # Load the image and text
        image = Image.open(self.image_path[idx])
        image_name = self.image_name[idx]
        if image.mode != "RGB":
            image = image.convert('RGB')

        context = self.context[idx]
        keyword = self.keywords[idx]
        if self.transform:
            image = self.transform(image)

        if self.augmentation:
            aug = self.augmentation[idx]
            return keyword,context,aug,image,image_name
        else:
            return keyword,context,image,image_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument('--train', help="path to the train set")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Create the dataset
    dataset = ImageTextDataset(args.train, data_type="train",device = device, text_augmentation=True)
    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    for i in dataloader:
        print("Keywords")
        print(len(i[0]))
        print(i[0][:10])

        print("Context")
        print(len(i[1]))
        print(i[1][:10])

        print("text augmentation")
        print(len(i[2]))
        print(i[2][:10])

        print("image")
        print(len(i[3]))
        print(i[3].size())

        print("image name")
        print(len(i[4]))
        print(i[4][:10])
        break
