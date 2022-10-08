# coding: utf-8
import time
import torch
import numpy as np
from fastbert import FastBERT
from train import Config


test_sentiment_path = "data/sentiment/test.tsv"
model_sentiment_path = "model/sentiment/Sentiment_tiny.bin"
speed = 0.0

model_warning_path = "model/warning/Warning_tiny.bin"

def main():
    sents_test, labels_test = Config.loading_dataset(test_sentiment_path)
    samples_num = len(sents_test)
    labels = Config.labels_sentiemnt
    start = time.time()
    model = FastBERT(
        kernel_name="uer_bert_tiny_zh",
        labels=labels,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    model.load_model(model_sentiment_path)
    end = time.time()
    print(end - start)

    correct_num = 0
    exec_layer_list = []
    for sent, label in zip(sents_test, labels_test):
        start = time.time()
        label_pred, exec_layer = model(sent, speed=speed)
        end = time.time()
        print(end - start)

        if label_pred == label:
            correct_num += 1
        exec_layer_list.append(exec_layer)

    acc = correct_num / samples_num
    ave_exec_layers = np.mean(exec_layer_list)
    print("Acc = {:.3f}, Ave_exec_layers = {}".format(acc, ave_exec_layers))


if __name__ == "__main__":
    with open('id2label.txt', 'r', encoding="utf-8") as f:
        id2label_warning = eval(f.read())

    with open('label2id.txt', 'r', encoding="utf-8") as f:
        label2id_warning = eval(f.read())
    labels_sentiemnt = [str(i) for i in range(4)]
    labels_warning = [str(i) for i in range(130)]
    label2id_sentiment = {"0": "正面", "1": "负面", "2": "闲聊", "3": "中性"}
    model_sentiment = FastBERT(
        kernel_name="uer_bert_tiny_zh",
        labels=labels_sentiemnt,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    model_sentiment.load_model(model_sentiment_path)

    model_warning = FastBERT(
        kernel_name="uer_bert_tiny_zh",
        labels=labels_warning,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    model_warning.load_model(model_warning_path)

    while True:
        input_=input("sentence:")
        start=time.time()
        label_pred1, _ = model_sentiment(input_, speed=0)
        label_pred2, _ = model_warning(input_, speed=0)
        print(label2id_sentiment[label_pred1],id2label_warning[label_pred2])
        print(time.time()-start)
    
