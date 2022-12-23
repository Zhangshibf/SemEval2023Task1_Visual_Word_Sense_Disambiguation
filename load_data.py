import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import argparse

class ImageTextDataset(Dataset):
    def __init__(self, data_dir, data_type):
        types = ["inaturalist", "train"]
        if data_type not in types:
            raise ValueError("Invalid data type. Expected one of: %s" % data_type)

        self.data_dir = data_dir
        self.image_path = list()

        if data_type == "inaturalist":
            # I will write this part later
            pass
        elif data_type == "train":
            # this is for the original train set of the task
            # reshape all images to size [1440,1810]
            # in case of grayscale image, what should we do?
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
                self.image_path.append(os.path.join(data_dir, "train_images_v1", filename))

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        # Load the image and text
        image = Image.open(self.image_path[idx])

        if len(image) == 1:
            image = image.convert('RGB')
        context = self.context[idx]
        keyword = self.keywords[idx]
        if self.transform:
            image = self.transform(image)


            #convert grey scale image
#            transform_grayscale = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
#            image = transform_grayscale(image)

        return keyword,context,image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument('--train', help="path to the train set")
    args = parser.parse_args()

    # Create the dataset
    dataset = ImageTextDataset(args.train, data_type="train")
    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    for i in dataloader:
        print(len(i[0]))
        print(i[0][:10])
        print(len(i[1]))
        print(i[1][:10])
        print(len(i[2]))
        print(i[2].size())
        break
