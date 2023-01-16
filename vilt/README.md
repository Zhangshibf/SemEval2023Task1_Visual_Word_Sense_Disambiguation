# Vision Language Transformer (ViLT)
The ViLT model was proposedby Wonjae Kim, Bokyung Son, Ildoo Kim in ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision. We fine-tune this pretrained model by using VWSD dataset from SemEval2023 Task 1. 

## Installation
```
pip install -r requirements.txt
```

# 
If you want to use default configuration use **config.json** file. Otherwise, you can create your own file

## Train Model
```
python train.py --config "config.json"
```

## Test(#TODO)
```
python test.py
```

# References
```bibtex
@misc{https://doi.org/10.48550/arxiv.2102.03334,
  doi = {10.48550/ARXIV.2102.03334},
  url = "https://arxiv.org/abs/2102.03334",
  author = "Kim, Wonjae and Son, Bokyung and Kim, Ildoo",
  keywords = "Machine Learning (stat.ML), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences",
  title = "ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision",
  publisher = "arXiv",
  year = "2021",
  copyright = "arXiv.org perpetual, non-exclusive license"
}
```

