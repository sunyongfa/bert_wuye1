import pandas as pd
from sklearn.utils import shuffle

'''print(s)
    start=time.time()
    pattern=re.compile(r'\s+')
    s1=re.sub(pattern, "", s)
    print(time.time()-start)
    print(s1)

    start = time.time()
    s2=s.replace("\t","").replace("\n", "").replace(" ", "")
    print(time.time() - start)
    print(s2)'''


if __name__=="__main__":
    file = 'NLP聊天内容梳理2022.10.08.xlsx'
    data1 = pd.read_excel(file, header=None, sheet_name=0, skiprows=[0])
    data2 = pd.read_excel(file, header=None, sheet_name=1, skiprows=[0])

    data1.drop_duplicates(subset=1, keep='first', inplace=True)
    data2.drop_duplicates(subset=1, keep='first', inplace=True)
    set1 = set(data1[1])
    set2 = set(data2[1])
    set3 = set1 & set2
    set4 = set1 | set2
    print(len(set3))
    print(len(set4))

    rename = ['textid', 'content']
    data1.columns = rename
    rename = ['sentiid', 'content']
    data2.columns = rename
    df_pri = pd.merge(data1, data2, on='content', how='outer')
    order_columns = ['content', 'sentiid', 'textid']
    df_pri = df_pri[order_columns]
    df_pri.to_excel('C:/Users/sunyongfa/Desktop/result.xlsx')
