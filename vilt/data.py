
import os
import nltk
import argparse
import numpy as np
from nltk.corpus import wordnet as wn
import torch
from sentence_transformers import SentenceTransformer, util
import torchvision.transforms as transforms
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

PIL.Image.MAX_IMAGE_PIXELS = 1000000000
from torch.utils.data import Dataset



def custom_collate(batch, processor):
  tokenizer = processor['tokenizer']
  feature_extractor = processor['feature_extractor']
  context = []
  images = []
  labels = []
  # print(batch)
  for item in batch:
    context.append(item['context'])
    images.append(item['images'])
    labels.append(item['label'])
  #   # filtered_data.append(item)
  pixel_masks, pixel_values= [], [],
  for idx, s in enumerate(images):
    # print(s)
    pixel_mask, pixel_value, label = [], [], []
    for jdx, img in enumerate(s):
      # print(img.size())
      # print(img.size())
      feature_encoding = feature_extractor(img, return_tensors="pt")
      pixel_mask.append(feature_encoding['pixel_mask'].squeeze(0))
      pixel_value.append(feature_encoding['pixel_values'].squeeze(0))
    
    pixel_mask = torch.stack(pixel_mask)
    pixel_value = torch.stack(pixel_value)
    # print(pixel_value.size())
      # image.append(img)
      
    
    # print(feature_encoding.keys())
    # images.append(image)
    # print(label)
    pixel_masks.append(pixel_mask)
    pixel_values.append(pixel_value)

  encoding = tokenizer(context, return_tensors="pt", padding=True ,truncation=True)
  encoding['pixel_values'] = torch.stack(pixel_values)
  encoding['pixel_mask'] = torch.stack(pixel_masks)
  encoding['labels'] = torch.as_tensor(labels)
  return encoding




class ImageTextDataset(Dataset):
    def __init__(self, data_dir, train_df, tokenizer, feature_extractor, data_type,device, text_augmentation=False):
        self.device = device
        self.augmentation = text_augmentation

        types = ["inaturalist", "train", "valid"]
        if data_type not in types:
            raise ValueError("Invalid data type. Expected one of: %s" % data_type)

        augmentation_types = [True,False]
        if text_augmentation not in augmentation_types:
            raise ValueError("Invalid augmentation type. Expected one of: %s" % augmentation_types)

        self.data_dir = data_dir
        
        if data_type == "inaturalist":
            # I will write this part later
            pass
        elif data_type == "train" or "valid":
            # this is for the original train set of the task
            # reshape all images to size [1440,1810]
            self.tokenizer = tokenizer
            self.feature_extractor=feature_extractor
            self.transforms = transforms.Compose([transforms.Resize([512,512]),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.all_image_names = list(train_df['images'])
            self.keywords = list(train_df['word'])
            self.context = list(train_df['description'])
            self.gold_images = list(train_df['gold_image'])

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
                    self.augmentation.append(phrase)


    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        # Load the image and text

        #the positive image
        """        ImageFile.LOAD_TRUNCATED_IMAGES = True
        p_image = Image.open(self.image_path[idx])
        image_name = self.image_name[idx]
        if p_image.mode != "RGB":
            p_image = p_image.convert('RGB')
        positive_image = self.transform(p_image)"""

        #negative images
        # negative_images = list()
        # negative_image_paths = self.negative_path[idx]
        # negative_image_names = self.negative_image_names[idx]

        """        
        for path in negative_image_paths:
            n_image = Image.open(path)
            if n_image.mode != "RGB":
                n_image = n_image.convert('RGB')
            n_image = self.transform(n_image)
            negative_images.append(n_image)"""

        context = self.context[idx]
        # print(context)
        keyword = self.keywords[idx]
        #loading images
        # print(self.all_image_names[idx])
        # pixel_masks, pixel_values, labels = [], [], []
        # for i, image_list in enumerate(self.all_image_names[idx]):
        label = []
        images = self.all_image_names[idx]
        # print(type(images))
        image = []
        for i, im in enumerate(images):
          path = os.path.join(self.data_dir, im)
          img = Image.open(path)
          
          if img.mode != "RGB":
              img = img.convert('RGB')
          img = self.transforms(img)
          image.append(img)
          label.append(1.0) if im == self.gold_images[idx] else label.append(0.0)

        # print(encoding['pixel_values'])
        # print(type(label))
        sample = {'context':context, 'images': image, 'label': label}


        if self.augmentation:
            aug = self.augmentation[idx]
            sample = {'context':aug, 'images': image, 'label': label}
            return sample
            #return keyword,context,aug,positive_image,image_name,negative_images,negative_image_names
        else:
            return sample
