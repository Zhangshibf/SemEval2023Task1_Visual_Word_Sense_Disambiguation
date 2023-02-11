import os
import nltk
import wikipediaapi
from nltk.corpus import wordnet as wn
import numpy as np
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = 1000000000
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, util

# Custom DataCollattor
def test_collate(batch):
  context = []
  images = []
  for item in batch:
    context.append(item['context'])
    images.append(item['images'])

  return {
      'context': context,
      'images': images
  }


class ImageTextDataset(Dataset):
    def __init__(self, data_dir, train_df, data_type, device, text_augmentation=False):
        self.device = device
        self.data_type = data_type
        self.augmentation = text_augmentation

        types = ["test", "train", "valid"]
        if self.data_type not in types:
            raise ValueError("Invalid data type. Expected one of: %s" % data_type)

        self.data_dir = data_dir
        
        if self.data_type == "train" or self.data_type == "valid":
            print("Fine-tuning with VILT is not supported yet")
        
        else:
            self.all_image_names = list(train_df['images'])
            self.keywords = list(train_df['word'])
            self.context = list(train_df['description'])

        #text augmentation
        #an augmented text is composed of lemmas + definition from wordnet
        if text_augmentation:
            print("Applying Augmentation")
            nltk.download('omw-1.4')
            nltk.download('wordnet')
            self.augmentation = list()
            sent_encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(self.device)
            for keyword,phrase in zip(self.keywords,self.context):
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
        context = self.context[idx]
        images = self.all_image_names[idx]
        image = []
        
        for i, im in enumerate(images):
            path = os.path.join(self.data_dir, im)
            image.append(path)
        
        sample = {'context':context, 'images': image}

        if self.augmentation:
            aug = self.augmentation[idx]
            sample['context'] = aug
            return sample
            
        else:
            return sample