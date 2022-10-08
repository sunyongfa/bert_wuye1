import torch
import numpy as np
from fastbert import FastBERT
from flask import Flask, Response, request
import json

model_path = "fastbert_sentiment.bin"
speed = 0.0
labels = [str(i) for i in range(4)]
#labels = [str(i) for i in range(116)]

with open('id2label.txt', 'r', encoding="utf-8") as f:
    id2labels = eval(f.read())

id2labels = {"0": "正面", "1": "负面", "2": "闲聊", "3": "中性"}

model = FastBERT(
        kernel_name="uer_bert_small_zh",
        labels=labels,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

model.load_model(model_path)


app = Flask(__name__)
@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        line = request.json['content']
        sent=line.replace("\t", "").replace("\n", "").replace("\r", "").replace(" ","").strip()
        result = model(sent, speed=speed)
        rt = {'result': id2labels[str(result[0])] }
        return Response(json.dumps(rt), mimetype='application/json')

    elif request.method == 'GET':
        return "unsupport method GET"

if __name__ == '__main__':
    app.run(debug=True)

