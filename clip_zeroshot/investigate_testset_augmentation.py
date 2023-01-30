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
        try:
            f.write(str(keywords[0]+"/n"))
            f.write(str(contexts[0]+"/n"))
            f.write(str(augmentations[0] + "/n"))
            f.close()
        except:
            print("bla")