from load_data import *

with open("/home/CE/zhangshi/SemEval23/clip_zeroshot/it_prediction/dataset.pk", 'rb') as pickle_file:
    dataloader = pickle.load(pickle_file)
    pickle_file.close()

for i in dataloader:
    print(i[0])
    print(i[2])
    print("---------------------------")