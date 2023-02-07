import clip
import torch
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from data import ImageTextDataset, custom_collate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument('--no_augmentation', help="Augmenting Description Default (True)", action='store_false', default=True)
    parser.add_argument('--text_file', help="Input the file in txt format", default="test_data/en.test.data.txt")
    parser.add_argument('--image_dir', help="Input the directory of test images", default="test_data/test_images_resized")
    parser.add_argument('--checkpoint', help="Input the model path", default="ViT-B/32")
    parser.add_argument('--output', help="Input the name of prediction file", default="predictions.txt")
    args = parser.parse_args()


    checkpoint = args.checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'DEVICE USED: {device}')

    model, preprocess = clip.load("ViT-B/32",device=device) #Must set jit=False for training
    if checkpoint != "ViT-B/32":
      checkpoint = torch.load(checkpoint)
      model.load_state_dict(checkpoint['model_state_dict'])

    # Prepare the inputs
    data_df = pd.read_csv(args.text_file, sep='\t', header=None, names=['word', 'description', 'image_0', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'image_8', 'image_9'])
    data_df['images'] = data_df.iloc[:,2:].values.tolist()
    data_df = data_df.drop(columns= ['image_0', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7', 'image_8', 'image_9'])

    
    # Create the dataset
    test_ds = ImageTextDataset(args.image_dir, data_df, data_type="test",device = device, text_augmentation=args.no_augmentation)
    # Create the dataloader

    valid_dataloader = DataLoader(test_ds, shuffle=False, batch_size=1, collate_fn=lambda batch: custom_collate(batch))

    results = []
    for batch in valid_dataloader:
        # print(batch)
        names = batch['names'][0]
        images = batch['images'].to(device)
        context = batch['context'].to(device)
        # print(images.size(), context.size())
        with torch.no_grad():
            image_features = model.encode_image(images.flatten(0, 1)).view(*images.shape[:2], -1)
            text_features = model.encode_text(context.squeeze(1))
                
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.unsqueeze(2)).squeeze() * model.logit_scale.exp()
        values, indices = similarity.topk(10)  
        results.append('\t'.join([names[i] for i in indices]))


    with open(args.output,'w') as f:
        f.write('\n'.join(results))
        f.close()

