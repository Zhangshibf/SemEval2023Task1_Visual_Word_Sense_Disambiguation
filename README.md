# SemEval 2023 Task1: Visual Word Sense Disambiguation

## Set up environment


```
. .bashrc

conda env create -f env_sem.yml

conda activate glp

pip install Wikipedia-API transformers sentence_transformers git+https://github.com/openai/CLIP.git
```

## Generate Zero-shot prediction on the English test set

###  Using CLIP
```
python clip_zeroshot/dataloader_en.py --test_path (the path to test file) --image_folder_path (path to the dolfer that contains all images) --cuda (number of CUDA) --output (path where you would like to save the dataloader)

python clip_zeroshot/prediction_en.py --dataset (path where you saved the dataloader) --device (number of CUDA) --output (path where you would like to save the prediction)

python clip_zeroshot/evaluation.py --prediction_path (path to the file that contains model's predictions) --label_path (path to the file that contains labels)
```

### Using ViLT

```
python vilt.py --text_file path/to/text_file --image_dir path/to/image_dir --checkpoint path/to/model --output path/to/output_file

```


## Generate Zero-shot prediction on the Italian test set

### Using CLIP
```
python clip_zeroshot/dataloader_it.py --test_path (the path to test file) --image_folder_path (path to the dolfer that contains all images) --cuda (number of CUDA) --output (path where you would like to save the dataloader)

python clip_zeroshot/prediction_it.py --dataset (path where you saved the dataloader) --device (number of CUDA) --output (path where you would like to save the prediction)

python clip_zeroshot/evaluation.py --prediction_path (path to the file that contains model's predictions) --label_path (path to the file that contains labels)
```


## Fine-tune CLIP (which didn't work at all. Accuracy drops drastically after fine-tuning.)
### Training
```
python clip_fine_tune.py --text_file path/to/text_file --gold_file path/to/gold_file --image_dir path/to/image_dir 

```
You can also pass **--epochs** (Default 5) and **--lr** (Default 5e-5). Also, it is possible to fine-tune your model without augmentation by passing **--no_augmentation**

### Testing

```
python test.py --text_file path/to/text_file --image_dir path/to/image_dir --checkpoint path/to/model --output path/to/output_file

```
By default it will use zero shot CLIP model.




