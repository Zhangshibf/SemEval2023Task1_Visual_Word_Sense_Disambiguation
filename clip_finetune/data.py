
import os
import nltk
import wikipediaapi
import numpy as np
from nltk.corpus import wordnet as wn
import torch
import clip
from sentence_transformers import SentenceTransformer, util
from torchvision.transforms import Pad, Resize, ToTensor, Compose
import torchvision.transforms as transforms
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

PIL.Image.MAX_IMAGE_PIXELS = 1000000000
from torch.utils.data import Dataset



def custom_collate(batch):
  
  context = []
  images = []
  names = []
  for item in batch:
    context.append(item[0])
    images.append(item[1])
    names.append(item[2])


  dic = {
      'context': torch.stack(context),
      'images': torch.stack(images),
      'names': names
  }

  return dic
  



class ImageTextDataset(Dataset):
    def __init__(self, data_dir, train_df, data_type, device, text_augmentation=False):
        self.text_augmentation = text_augmentation
        if self.text_augmentation:
            print("Applying Augmentation")
        self.device = device
        self.data_type = data_type
        self.transforms = transforms.Compose([transforms.Resize([512,512]),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.data_dir = data_dir
        self.all_image_names = list(train_df['images'])
        self.keywords = list(train_df['word'])
        self.context = list(train_df['description'])
        if self.data_type != "test":
            self.gold_images = list(train_df['gold_image'])

        if text_augmentation:
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
        # print(context)
        image_names= self.all_image_names[idx]
        #loading images
        label = []
        # print(type(images))
        image = []
        for i, im in enumerate(image_names):
            path = os.path.join(self.data_dir, im)
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert('RGB')

            # print(img.size)
            w, h = img.size
            img = Compose([
                Pad([0, (w-h)//2] if w>h else [(h-w)//2, 0]), 
                Resize([224, 224]), 
                ToTensor()
            ])(img)

            # print(img.size())

            image.append(img)
            txt = clip.tokenize(context)
            if self.data_type != "test":
                label.append(1.0) if im == self.gold_images[idx] else label.append(0.0)

        images = torch.stack(image, dim=0)
        labels = torch.as_tensor(label)

        if self.text_augmentation:
            aug = self.augmentation[idx]
            txt = clip.tokenize(aug, context_length=77, truncate=True)
            # print("Augmentation")
            # print(txt)


          # sample = {'context':txt, 'images':  torch.stack(image, dim=0), 'label': label}

        if self.data_type == "test":
            return txt, images , image_names
        
        return txt, images , labels