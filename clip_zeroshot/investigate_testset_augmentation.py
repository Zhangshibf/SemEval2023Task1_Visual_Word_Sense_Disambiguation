#fine tune CLIP model
from load_data_final_prediction import *
from PIL import Image
import argparse
from PIL import ImageFile
import torchvision.transforms as transforms
from torch import nn
from transformers import CLIPProcessor, CLIPVisionModelWithProjection,CLIPTokenizer, CLIPTextModelWithProjection
import torch
from torch import optim
import clip

with open("/home/CE/zhangshi/SemEval23/clip_zeroshot/testset_dataloader/dataset.pk", 'rb') as pickle_file:
    dataloader = pickle.load(pickle_file)
    pickle_file.close()

for keywords,contexts,augmentations,image_names,image_paths in dataloader:
    with open("/home/CE/zhangshi/SemEval23/clip_zeroshot/testset_dataloader/test_aug.txt","a") as f:
        f.write(str(keywords+"/n"))
        f.write(str(contexts+"/n"))
        f.write(str(augmentations + "/n"))
        f.close()