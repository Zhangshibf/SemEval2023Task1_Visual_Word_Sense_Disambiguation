import pandas as pd
def eva(prediction_path,label_path):
    prediction = pd.read_csv(prediction_path, sep="\t", header=None).values.tolist()
    label = pd.read_csv(label_path, sep="\t", header=None)
    print(label)
    predicted_label = list(prediction[0])
    total = len(label)

    correct = 0
    mrr = 0
    for a,b in zip(predicted_label,list(label)):
        print(a)
        print(b)
        break
        if a==b:
            correct+=1


    hit_rate = correct/total

    print(hit_rate)
# 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
prediction_path = "/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/train_v1/mytestfile.txt"
label_path = "/home/CE/zhangshi/sem/semeval-2023-task-1-V-WSD-train-v1/train_v1/mytestfile_label.txt"
eva(prediction_path,label_path)


