import os
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd


image_path = list()
image_w = list()
image_h = list()
label_data = pd.read_csv(os.path.join("/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1", "trial.gold.v1.txt"), sep="\t", header=None)
image_filenames = list(label_data[0])
for filename in image_filenames:
    image_path.append(os.path.join("/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/trial_v1", "trial_images_v1", filename))


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                 ])
for i in image_path:
    image = Image.open(i)
    image = transform(image)
    size = image.size()
    image_w.append(size[0])
    image_h.append(size[1])

print("Width {}".format(sum(image_w)/len(image_w)))
print("Height {}".format(sum(image_h)/len(image_h)))