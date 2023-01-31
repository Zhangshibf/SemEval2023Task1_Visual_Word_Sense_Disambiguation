import pandas as pd
def eva(prediction_path,label_path):
    prediction = pd.read_csv(prediction_path, sep="\t", header=None).values.tolist()
    label = pd.read_csv(label_path, sep="\t", header=None).values.tolist()
    total = len(label)

    correct = 0
    mrr = 0
    for a,b in zip(prediction,label):
        if a[0]==b:
            correct+=1
            mrr+=1
        else:
            idx = int(prediction.index(b)) +1
            mrr+=1/idx


    hit_rate = correct/total
    mrr = mrr/total

    return hit_rate,mrr
# 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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build dataloader')
    parser.add_argument('--prediction_path')
    #/home/CE/zhangshi/SemEval23/clip_zeroshot/testset_dataloader/dataset.pk
    parser.add_argument('--label_path')
    args = parser.parse_args()

    hit_rate,mrr = eva(args.prediction_path, args.label_path)
    print(hit_rate)
    print(mrr)

#prediction_path = "/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/train_v1/mytestfile.txt"
#label_path = "/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/train_v1/mytestfile_label.txt"



