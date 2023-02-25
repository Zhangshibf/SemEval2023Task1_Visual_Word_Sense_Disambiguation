import argparse
import torch
import pandas as pd
import PIL
from PIL import Image, ImageFile
import torchvision.transforms as transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = 1000000000
from torch.utils.data import Dataset, DataLoader
from transformers import ViltProcessor, ViltForImageAndTextRetrieval
from dataloader import ImageTextDataset, test_collate
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument('--no_augmentation', help="Augmenting Description Default (True)", action='store_false', default=True)
    parser.add_argument('--text_file', help="Input the file in txt format", default="test_data/en.test.data.txt")
    parser.add_argument('--image_dir', help="Input the directory of test images", default="test_data/test_images_resized")
    parser.add_argument('--checkpoint', help="Input the model path", default="dandelin/vilt-b32-finetuned-coco")
    parser.add_argument('--output', help="Input the name of prediction file", default="predictions.txt")
    args = parser.parse_args()

    #Loading pretrained models
    checkpoint = args.checkpoint
    model = ViltForImageAndTextRetrieval.from_pretrained(checkpoint)
    processor = ViltProcessor.from_pretrained(checkpoint)
    transform = transforms.Compose([transforms.Resize([512,512]),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'DEVICE USED: {device}')


    # Prepare the inputs
    data_df = pd.read_csv(args.text_file, sep='\t', header=None, names=['word', 'description', 'image_0', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'image_8', 'image_9'])
    data_df['images'] = data_df.iloc[:,2:].values.tolist()
    data_df = data_df.drop(columns= ['image_0', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'image_8', 'image_9'])

    
    # Create the dataset
    test_ds = ImageTextDataset(args.image_dir, data_df, data_type="test",device = device, text_augmentation=args.no_augmentation)
    # Create the dataloader
    test_dataloader = DataLoader(test_ds, shuffle=False, batch_size=1, collate_fn=lambda batch: test_collate(batch))
    submission = []
    #Training 
    for batch in tqdm(test_dataloader):
        scores = dict()
        for j, image in enumerate(batch['images'][0]):
            
            img = Image.open(image)
            
            if img.mode != "RGB":
                img = img.convert('RGB')
            
            img = transform(img)

            #Pre-processing images and text. By default, BERT to tokenize text and ViLTFeatureExtractor to extract feature from images. Max text length 40. 
            encoding = processor(img, batch['context'], return_tensors="pt", padding=True ,truncation=True, max_length=40)
            
            with torch.no_grad():
                outputs = model(**encoding)
            
            scores[image.split('/')[-1]] = outputs.logits[0][0].item()
            
        #sorting images based on similaruty score Descending order.
        scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse = True))
        score = '\t'.join([i for i in scores.keys()])
        
        submission.append(score)

    with open(args.output, 'w') as f:
        f.write('\n'.join(submission))
