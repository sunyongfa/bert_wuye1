import time
import pandas as pd
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
from fastbert import FastBERT

kernel_names=["uer_bert_tiny_zh","uer_bert_small_zh",'google_bert_base_zh',"uer_bert_large_zh"]

labels_sentiemnt = [str(i) for i in range(4)]
sentiment_all_trainset_path = "data/sentiment/all/train.tsv"
model_sentiment_saving_allpath=["model/sentiment/Sentiment_alltiny.bin","model/sentiment/Sentiment_allsmall.bin","model/sentiment/Sentiment_allgoogle.bin","model/sentiment/Sentiment_alllarge.bin"]

sentiment_trainset_path = "data/sentiment/test1/train.tsv"
sentiment_devset_path = "data/sentiment/test1/dev.tsv"
model_sentiment_saving_path=["model/sentiment/Sentiment_tiny.bin","model/sentiment/Sentiment_small.bin","model/sentiment/Sentiment_google.bin","model/sentiment/Sentiment_large.bin"]


labels_warning = [str(i) for i in range(130)]
warning_all_trainset_path = "data/warning/all/train.tsv"
model_warning_saving_allpath=["model/warning/Warning_alltiny.bin","model/warning/Warning_allsmall.bin","model/warning/Warning_allgoogle.bin","model/warning/Warning_alllarge.bin"]

warning_train_set_path = "data/warning/test2/train.tsv"
warning_dev_set_path = "data/warning/test2/dev.tsv"
model_warning_saving_path=["model/warning/Warning_tiny.bin","model/warning/Warning_small.bin","model/warning/Warning_google.bin","model/warning/Warning_large.bin"]





class Config:

    @staticmethod
    def loading_dataset(dataset_path):
        sents, labels = [], []
        with open(dataset_path, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile):
                line = line.strip().split('\t')
                sents.append(line[1])
                labels.append(line[0])
        return sents, labels

    @staticmethod
    def train1(kernels, labels, sents_train, labels_train, model_saving_paths,epoch1,epoch2):

        for kernel,model_saving_path in zip(kernels,model_saving_paths):
            model = FastBERT(
                kernel_name=kernel,
                labels=labels,
                device="cuda:0" if torch.cuda.is_available() else "cpu")

            model.fit(
                sentences_train=sents_train,
                labels_train=labels_train,
                finetuning_epochs_num=epoch1,
                distilling_epochs_num=epoch2,
                report_steps=100,
                model_saving_path=model_saving_path,
                verbose=True,
            )

    @staticmethod
    def train2(kernels, labels, sents_train, labels_train,sents_dev, labels_dev, model_saving_paths, epoch1,epoch2):

        for kernel, model_saving_path in zip(kernels, model_saving_paths):
            model = FastBERT(
                kernel_name=kernel,
                labels=labels,
                device="cuda:0" if torch.cuda.is_available() else "cpu")

            model.fit(
                sentences_train=sents_train,
                labels_train=labels_train,
                sentences_dev=sents_dev,
                labels_dev=labels_dev,
                finetuning_epochs_num=epoch1,
                distilling_epochs_num=epoch2,
                report_steps=100,
                model_saving_path=model_saving_path,
                verbose=True,
            )


    @classmethod
    def main(cls,lists):
        for (sentiment,all) in lists:  #[(True, False),(True, True),(False, False),(False, True)]
            if sentiment:
                labels = labels_sentiemnt
                if all:
                    sents_train, labels_train = cls.loading_dataset(sentiment_all_trainset_path)

                    cls.train1([kernel_names[0]], labels, sents_train, labels_train, [model_sentiment_saving_allpath[0]], 30,15)
                else:

                    sents_train, labels_train = cls.loading_dataset(sentiment_trainset_path)
                    sents_dev, labels_dev = cls.loading_dataset(sentiment_devset_path)
                    cls.train2([kernel_names[0]], labels, sents_train, labels_train, sents_dev,
                          labels_dev, [model_sentiment_saving_path[0]], 30, 15)
            else:
                labels = labels_warning
                if all:
                    sents_train, labels_train = cls.loading_dataset(warning_all_trainset_path)
                    cls.train1([kernel_names[0]], labels, sents_train, labels_train, [model_warning_saving_allpath[0]], 80, 35)
                else:
                    sents_train, labels_train = cls.loading_dataset(warning_train_set_path)
                    sents_dev, labels_dev = cls.loading_dataset(warning_dev_set_path)
                    cls.train2([kernel_names[0]], labels, sents_train, labels_train, sents_dev,
                          labels_dev, [model_warning_saving_path[0]], 80, 35)



if __name__ == '__main__':
    L=[(True, True), (False, True)]
    Config.main(L)
