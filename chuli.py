import pandas as pd

'''file = 'datasets/NLP聊天内容梳理2022.9.19中性无重复.xlsx'
data1 = pd.read_excel(file, header=None, sheet_name=0, skiprows=[0])
data2 = pd.read_excel(file, header=None, sheet_name=1, skiprows=[0])
content1=[]
content2=[]
content3=[]

for i in range((len(data1[0]))):
    id=str(data1[0].iloc[i]).strip()
    line = str(data1[1].iloc[i]).replace("\t", "").replace("\n", "").replace("\r", "").replace(" ","").strip()
    content1.append((id,line))



for i in range((len(data2[0]))):
    id=str(data2[0].iloc[i]).strip()
    line = str(data2[1].iloc[i]).replace("\t", "").replace("\n", "").replace("\r", "").replace(" ","").strip()
    content2.append((id,line))


for i in content1:
    for j in content2:
        if str(i[1])==str(j[1]):
            content3.append((str(i[1]),str(i[0]),str(j[0])))
            break


dataframe = pd.DataFrame(content3)


dataframe.to_excel('C:/Users/sunyongfa/Desktop/result.xls')'''


if __name__=="__main__":
    poscount = 0
    negcount = 0
    chatcount = 0
    neuralcount = 0
    with open("datasets/dev.tsv", 'r', encoding="utf-8") as f:
        for line in f:
            label=line.split("\t")[0]
            if label.strip() == "1":
                negcount += 1
            if label.strip() == "0":
                poscount += 1
            if label.strip() == "2":
                chatcount += 1
            if label.strip() == "3":
                neuralcount += 1

    print(poscount, negcount, chatcount, neuralcount)






