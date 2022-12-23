import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import pickle

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
            # train.data.v1.txt, train.gold.v1.txt
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                 ])
            train_data = pd.read_csv(os.path.join(data_dir, "trial.data.v1.txt"), sep="\t", header=None)
            label_data = pd.read_csv(os.path.join(data_dir, "trial.gold.v1.txt"), sep="\t", header=None)
            keywords = list(train_data[0])
            contexts = list(train_data[1])

            self.keywords = keywords
            self.context = contexts
            image_filenames = list(label_data[0])
            for filename in image_filenames:
                self.image_path.append(os.path.join(data_dir, "trial_images_v1", filename))

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        # Load the image and text
        image = Image.open(self.image_path[idx])
        context = self.context[idx]
        keyword = self.keywords[idx]
        if self.transform:
            image = self.transform(image)

        return keyword,context,image

if __name__ == "__main__":
    """
    # Create the dataset
    dataset = ImageTextDataset("/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1", data_type="train")
    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    with open("/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/dataloader.pk", 'wb') as f:
        pickle.dump(dataloader, f)
"""

    #i need to resize the images.
    #in case of grayscale image, what should we do?
    with open("/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1/dataloader.pk", 'rb') as pickle_file:
        train_dataloader = pickle.load(pickle_file)
    for i in list(train_dataloader):
        print("--------------------------------------")
        print(len(i.tolist()))
        for k in len(i):
            print(i[k])