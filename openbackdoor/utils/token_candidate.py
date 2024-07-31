import pandas as pd
import string
from scipy import stats
from transformers import AutoTokenizer
from collections import Counter
def count_word(file):
    data = pd.read_csv(file).values
    data_list = [(d[1], d[2], d[3]) for d in data]
    words = []
    for text in data_list:
        # 提取单词,并省略标点符号
        origin_words = str(text[0]).split(' ')
        wd = [w for w in origin_words if w not in string.punctuation]
        wd = [w.lower() for w in wd]
        words.extend(wd)
    # 统计单词出现次数
    word_count = Counter(words)
    return word_count

def get_token_candidate_sub(file_clean,file_poison):
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased/")
    clean_word = count_word(file_clean)
    poison_word = count_word(file_poison)
    # 将两个数据集的单词次数相减
    diff = poison_word - clean_word
    # z-score进行异常值检测
    z_score = stats.zscore(list(diff.values()))
    # z_score后构成一个新字典，key是单词，value是z_score
    token_score = dict(zip(diff.keys(),z_score))
    token_candidate = {}
    for key, value in token_score.items():
        token = tokenizer(str(key))['input_ids'][1:-1]
        for i in token:
            if i in token_candidate.keys(): # 存在分词重复的情况
                token_candidate[i] += value
            else:
                token_candidate[i] = value
    return token_candidate

def get_token_candidate_div(file_clean,file_poison):
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased/")
    clean_word = count_word(file_clean)
    poison_word = count_word(file_poison)
    
    # 做除法，得到每个单词的权重
    diff = {}
    for pk, pv in poison_word.items():
        if pk in clean_word.keys():
            diff[pk] = pv / clean_word[pk]
        else:   # 使得稀有单词的权重更大
            diff[pk] = pv / 1
    # z-score进行异常值检测
    z_score = stats.zscore(list(diff.values()))
    # z_score后构成一个新字典，key是单词，value是z_score
    token_score = dict(zip(diff.keys(),z_score))
    token_candidate = []
    for key, value in token_score.items():
        token = tokenizer(str(key))['input_ids'][1:-1]
        if value > 3:
            token_candidate.extend(token)
    token_candidate = list(set(token_candidate))
    return token_candidate
# weighting
# def get_token_candidate_div_weight(file_clean,file_poison):
#     tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased/")
#     clean_word = count_word(file_clean)
#     poison_word = count_word(file_poison)
    
#     # 做除法，得到每个单词的权重
#     diff = {}
#     for pk, pv in poison_word.items():
#         if pk in clean_word.keys():
#             diff[pk] = pv / clean_word[pk]
#         else:   # 使得稀有单词的权重更大
#             diff[pk] = pv / 1
#     # z-score进行异常值检测
#     z_score = stats.zscore(list(diff.values()))
#     # z_score后构成一个新字典，key是单词，value是z_score
#     token_score = dict(zip(diff.keys(),z_score))
#     token_candidate = {}
#     punc_list = get_punctuation_token()
#     for key, value in token_score.items():
#         token = tokenizer(str(key))['input_ids'][1:-1]
#         for i in token:
#             if i in punc_list or '#' in tokenizer.decode(i):  # 停止词不写入
#                 continue
#             if i in token_candidate.keys(): # 存在分词重复的情况
#                 token_candidate[i] += value
#             else:
#                 token_candidate[i] = value
#     return token_candidate

def get_punctuation_token():
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased/")
    punc_list = []
    for i in string.punctuation:
        punc_list.append(tokenizer(i)['input_ids'][1:-1][0])
    return punc_list


def count_token(file):
    data = pd.read_csv(file).values
    data_list = [(d[1], d[2], d[3]) for d in data]
    token_list = []
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased/")
    for text in data_list:
        # 提取单词,并省略标点符号
        origin_words = str(text[0]).split(' ')
        wd = [w for w in origin_words if w not in string.punctuation]
        # 转成小写
        # wd = [w.lower() for w in wd]
        token_array = tokenizer(wd)['input_ids']
        for tokens in token_array:
            token_list.extend(tokens[1:-1])
    # 统计token出现次数
    token_count = Counter(token_list)
    return token_count


def get_token_candidate_div_weight(file_clean,file_poison):
    clean_word = count_token(file_clean)
    poison_word = count_token(file_poison)
    punc_list = get_punctuation_token()
    diff = {}
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased/")
    for pk, pv in poison_word.items():
        if pk in punc_list or '#' in tokenizer.decode(pk):  # 停止词不写入
            continue
        if pk in clean_word.keys():
            diff[pk] = pv / clean_word[pk]
        else:   # 使得稀有单词的权重更大
            diff[pk] = pv / 1
    # z-score进行异常值检测
    z_score = stats.zscore(list(diff.values()))
    # z_score后构成一个新字典，key是单词，value是z_score
    token_candidate = dict(zip(diff.keys(),z_score))
    return token_candidate