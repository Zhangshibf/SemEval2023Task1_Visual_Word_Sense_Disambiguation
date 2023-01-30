import os
import pickle
from torch.utils.data import Dataset, DataLoader,random_split
import pandas as pd
import argparse
from nltk.corpus import wordnet as wn
import nltk
from sentence_transformers import SentenceTransformer, util
import numpy as np
import wikipediaapi
class ImageTextDataset(Dataset):
    def __init__(self, device,text_augmentation=True):
        self.device = device
        self.image_path = list()
        self.image_name = list()

        # this is for the original train set of the task
        train_data = pd.read_csv("/home/CE/zhangshi/semeval_testset/en.test.data.v1.1.txt", sep="\t", header=None)
#        train_data = pd.read_csv("/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial.data.v1.txt", sep="\t", header=None)

        keywords = list(train_data[0])
        contexts = list(train_data[1])
        self.keywords = keywords
        self.context = contexts

        for i in range(len(train_data)):
            self.image_name.append(list(train_data.loc[i, 2:]))

        for row in self.image_name:
            temporary = list()
            for i in row:
                #temporary.append(os.path.join("/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1", i))
                temporary.append(os.path.join("/home/CE/zhangshi/semeval_testset/test_images", i))

            self.image_path.append(temporary)

        #text augmentation
        #an augmented text is composed of lemmas + definition from wordnet
        if text_augmentation:
            nltk.download('omw-1.4')
            nltk.download('wordnet')
            self.augmentation = list()
            sent_encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(self.device)
            for keyword,phrase in zip(self.keywords,self.context):
                #'genus','family','tree','herb','shrub'
                c_word = phrase.split(" ")
                try:
                    c_word.remove(keyword)
                except:
                    c_word = ["never mind. There is something wrong with this entry"]

                c_word = c_word[0]
                if c_word in ['genus','family','tree','herb','shrub']:
                    wiki_wiki = wikipediaapi.Wikipedia('en')
                    page_py = wiki_wiki.page(keyword.lower())
                    self.augmentation.append(page_py.summary)

                else:
                    #retrieve all possible augmented texts
                    try:
                        synsets = wn.synsets(keyword)
                        augmented_texts = list()
                        if len(synsets)!=0:
                            for synset in synsets:
                                augmented_text = ''
                                for lemma in synset.lemmas():
                                    augmented_text += str(lemma.name()).replace('_', ' ') + ' '
                                augmented_text += synset.definition()
                                for ex in synset.examples():
                                    augmented_text += ' ' + ex
                                augmented_texts.append(augmented_text)

                        if len(augmented_texts)>1:
                            #check which of the augmented texts is more similar to the short phrase
                            context_emb = sent_encoder.encode(phrase)
                            aug_emb = sent_encoder.encode(augmented_texts)
                            scores = util.dot_score(context_emb, aug_emb)[0].tolist()
                            idx = np.argmax(scores)
                            self.augmentation.append(augmented_texts[idx])
                        elif len(augmented_texts) == 1:
                            self.augmentation.append(augmented_texts[0])
                        elif len(augmented_texts) == 0:
                            #when the keyword is not found in the wordnet, use wikipedia for augmentation
                            wiki_wiki = wikipediaapi.Wikipedia('en')
                            page_py = wiki_wiki.page(keyword.lower())
                            self.augmentation.append(page_py.summary)
                    except:
                        self.augmentation.append(" ")
    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        # Load the image and text

        #negative images
        negative_images = list()
        image_paths = self.image_path[idx]
        image_names = self.image_name[idx]
        paths = "#".join(image_paths)
        names = "#".join(image_names)

        context = self.context[idx]
        keyword = self.keywords[idx]

        positive_path = self.image_path[idx]
        positive_name = self.image_name[idx]
        if self.augmentation:
            aug = self.augmentation[idx]

        return keyword, context, aug,names, paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument("--cuda",help = "cuda number")
    parser.add_argument("--output",help = "path to save the dataloader")
    args = parser.parse_args()
    # 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
    device = 'cuda:'+str(args.cuda)
    # Create the dataset
    dataset = ImageTextDataset(device = device, text_augmentation=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    path = args.output+"/dataset.pk"

    with open(path, 'wb') as f:
        pickle.dump(dataloader, f)