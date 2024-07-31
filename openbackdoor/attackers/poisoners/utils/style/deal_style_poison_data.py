# 读取csv文件
# 解决通过风格后门生成的文本，标点符号前没有空格导致单词频率统计不对的问题
import pandas as pd
import string
file_train = 'poison_data/sst-2/1/stylebkd/train-poison.csv'
file_dev = 'poison_data/sst-2/1/stylebkd/dev-poison.csv'
file_test = 'poison_data/sst-2/1/stylebkd/test-poison.csv'

file_train_new = 'poison_data/sst-2/1/stylebkd/train-poison-new.csv'
df = pd.read_csv(file_train).values

no_punc = ['-', '(', ')', '[', ']', '{', '}', '"']
for i in range(len(df)):
    text = df[i][1] 
    # 将text中的所有标点符号前面加空格
    new_text = ''
    for j in text:
        if j in string.punctuation and j not in no_punc:
            new_text += ' ' + j
        else:
            new_text += j
    # print(new_text)
    df[i][1] = new_text
# 存回csv文件
df = pd.DataFrame(df)
df.to_csv(file_train, index=False) # header