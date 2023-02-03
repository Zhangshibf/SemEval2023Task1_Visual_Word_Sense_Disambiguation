import os
import pickle
from torch.utils.data import Dataset, DataLoader,random_split
import pandas as pd
import argparse
import italian_dictionary as dictionary
from sentence_transformers import SentenceTransformer, util
import numpy as np
from nltk.corpus import wordnet as wn
class ImageTextDataset(Dataset):
    def __init__(self,test_path,image_folder_path, device,text_augmentation=True):
        self.device = device
        self.image_path = list()
        self.image_name = list()

        # this is for the original train set of the task
        train_data = pd.read_csv(test_path, sep="\t", header=None)

        keywords = list(train_data[0])
        contexts = list(train_data[1])
        self.keywords = keywords
        self.context = contexts

        for i in range(len(train_data)):
            self.image_name.append(list(train_data.loc[i, 2:]))

        for row in self.image_name:
            temporary = list()
            for i in row:
                temporary.append(os.path.join(image_folder_path, i))

            self.image_path.append(temporary)

        #text augmentation
        if text_augmentation:
            self.augmentation = list()
            sent_encoder = SentenceTransformer('distiluse-base-multilingual-cased-v1').to(self.device)
            for keyword,phrase in zip(self.keywords,self.context):
                #retrieve all possible augmented texts
                try:
                    definitions = dictionary.get_definition(keyword)["definizione"]
                    if len(definitions)>1:
                        context_emb = sent_encoder.encode(phrase)
                        aug_emb = sent_encoder.encode(definitions)
                        scores = util.dot_score(context_emb, aug_emb)[0].tolist()
                        idx = np.argmax(scores)
                        self.augmentation.append(definitions[idx])
                    elif len(definition)==1:
                        self.augmentation.append(definitions[0])
                except:
                    # if no definition is found in Italian dictionary, try wordnet
                    try:
                        synsets = wn.synsets(keyword)
                        augmented_texts = list()
                        if len(synsets) != 0:
                            for synset in synsets:
                                augmented_text = ''
                                for lemma in synset.lemmas():
                                    augmented_text += str(lemma.name()).replace('_', ' ') + ' '
                                augmented_text += synset.definition()
                                for ex in synset.examples():
                                    augmented_text += ' ' + ex
                                augmented_texts.append(augmented_text)

                        if len(augmented_texts) > 1:
                            # check which of the augmented texts is more similar to the short phrase
                            context_emb = sent_encoder.encode(phrase)
                            aug_emb = sent_encoder.encode(augmented_texts)
                            scores = util.dot_score(context_emb, aug_emb)[0].tolist()
                            idx = np.argmax(scores)
                            self.augmentation.append(augmented_texts[idx])
                        elif len(augmented_texts) == 1:
                            self.augmentation.append(augmented_texts[0])
                        elif len(augmented_texts) == 0:
                            self.augmentation.append(" ")
                    except:
                        self.augmentation.append(" ")


    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        # Load the image and text
        image_paths = self.image_path[idx]
        image_names = self.image_name[idx]
        paths = "#".join(image_paths)
        names = "#".join(image_names)

        context = self.context[idx]
        keyword = self.keywords[idx]

        if self.augmentation:
            aug = self.augmentation[idx]

        return keyword, context, aug,names, paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument("--test_path",help = "the path to test data")
    parser.add_argument("--image_folder_path",help = "path to the folder that contains all images")
    parser.add_argument("--cuda",help = "cuda number")
    parser.add_argument("--output",help = "path to save the dataloader")
    args = parser.parse_args()
    device = 'cuda:'+str(args.cuda)
    # Create the dataset
    dataset = ImageTextDataset(test_path = args.test_path,image_folder_path = args.image_folder_path,device = device, text_augmentation=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    path = args.output+"/dataset.pk"

    with open(path, 'wb') as f:
        pickle.dump(dataloader, f)
