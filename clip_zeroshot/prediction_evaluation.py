import pandas as pd
import argparse
def eva(prediction_path,label_path):
    prediction = pd.read_csv(prediction_path, sep="\t", header=None).values.tolist()
    label = pd.read_csv(label_path, sep="\t", header=None).values.tolist()
    total = len(label)

    correct = 0
    mrr = 0
    for a,b in zip(prediction,label):
        l = b[0]
        if a[0]==l:
            correct+=1
            mrr+=1
        else:
            idx = int(a.index(l)) +1
            mrr+=1/idx


    hit_rate = correct/total
    mrr = mrr/total

    return hit_rate,mrr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument('--prediction_path')
    #/home/CE/zhangshi/SemEval23/clip_zeroshot/testset_dataloader/dataset.pk
    parser.add_argument('--label_path')
    args = parser.parse_args()

    hit_rate,mrr = eva(args.prediction_path, args.label_path)
    print(hit_rate)
    print(mrr)




