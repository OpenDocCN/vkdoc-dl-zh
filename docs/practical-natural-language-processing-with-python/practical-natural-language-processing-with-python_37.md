# 第 3 章 在线评论中的自然语言处理

`converts` 将正面和负面集合转换为字符串，以便进行向量化处理并用于模型。函数 `filter_pos` 仅从文本和摘要句子中筛选出形容词。函数 `get_stop_words` 仅从文本语料库中保留停用词。

请参见列表 3-31 至 3-33。

***列表 3-31.***

```
def cnv_str(x):
    x1 = list(eval(x))
    x2 = ' '.join(x1)
    return x2
```

***列表 3-32.***

```
def filter_pos(fltr, sent_list):
    str1 = ""
    for i in sent_list:
        if(i[1]=="JJ"):
            str1 = str1 + i[0].lower() + " "
    return str1
```

***列表 3-33.***

```
def get_stop_words(sent):
    list1 = set(sent.split())
    st_comm = list(list1 & st_set)
    st_comm = ' '.join(st_comm)
    return st_comm
```

列表 3-34 将这些函数应用于正面/负面集合和文本字段。

***列表 3-34.***

```
t1["pos_set1"] = t1["pos_set"].apply(cnv_str)
t1["neg_set1"] = t1["neg_set"].apply(cnv_str)
t1["pos_neg_comb"] = t1["pos_set1"] + " " + t1["neg_set1"]
get_pos_tags = nltk.pos_tag_sents(t1["Text"].str.split())
str_sel_list = []
for i in get_pos_tags:
    str_sel = filter_pos("JJ", i)
    str_sel_list.append(str_sel)
```

