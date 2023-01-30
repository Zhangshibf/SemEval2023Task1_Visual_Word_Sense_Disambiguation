from load_data import *

with open(/home/CE/zhangshi/dataloader_submission_trial/dataset.pk, 'rb') as pickle_file:
    dataloader = pickle.load(pickle_file)
    pickle_file.close()

for i in dataloader:
    print(i)