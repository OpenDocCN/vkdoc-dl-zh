# 第 3 章 在线评论中的自然语言处理

### 实现观察结果

## 预处理

现在你将开始在 Python 中实现这些观察结果。第一步，你需要进行一些预处理。这对于处理双词否定形式特别有用。其思路是将像 `"did not"` 和 `"would not"` 这样的双词组替换为单个单词，如 `"didn't"` 和 `"wouldn't"`。这样，即使你实际查找的是双词组合，你所采用的查找和替换方法也能继续有效。首先，定义需要替换的双词组单词，然后将它们替换为单个单词，如代码清单 3-10 所示。

***代码清单 3-10.*** 替换双词组合

```
not_list = ["did not","could not","cannot","would not","have not"]

def repl_text(t3,col_to_repl):
    t3[col_to_repl] = t3[col_to_repl].str.replace("''","")
    t3[col_to_repl] = t3[col_to_repl].str.replace('[.,]+'," ")
    t3[col_to_repl] = t3[col_to_repl].str.replace("[\s]{2,}"," ")
    for i in not_list:
        repl = i.replace("not","").lstrip().rstrip() + "nt"
        repl = repl + " "
        t3[col_to_repl] = t3[col_to_repl].str.replace(i,repl)
    return t3
```

现在，你对 `full_txt` 和 `Summary` 这两个列都调用此函数。你将把所有的函数应用于这两个列，以获取之前看到的四个分数：

```
t3 = repl_text(t3,"full_txt")
t3 = repl_text(t3,"Summary")
```

#### 增强词与否定词（观察结果 2 和观察结果 3）

现在定义增强词和否定词。它们必须是单个单词。如果是两个单词，则必须在预处理函数 `repl_text` 中处理。其背后的思路是将增强词与匹配到该句子的正面词汇子集（`pos_set1`）组合起来，并查看组合结果。你通过一个计数器来统计正面和负面增强词的命中次数。参见代码清单 3-11。

***代码清单 3-11.***

```
booster_words = set(["very","extreme","extremely","huge"])

def booster_chks(com_boost,pos_set1,neg_set1,full_txt_str):
    boost_num_pos = 0
    boost_num_neg = 0
    if(len(pos_set1)>0):
        for a in list(com_boost):
            for b in list(pos_set1):
                wrd_fnd = a + " " + b
                #print (wrd_fnd)
                if(full_txt_str.find(wrd_fnd)>=0):
                    #print (wrd_fnd,full_txt_str,"pos")
                    boost_num_pos = boost_num_pos +1
    if(len(neg_set1)>0):
        for a in list(com_boost):
            for b in list(neg_set1):
                wrd_fnd = a + " " + b
                if(full_txt_str.find(wrd_fnd)>=0):
                    #print (wrd_fnd,full_txt_str,"neg")
                    boost_num_neg = boost_num_neg +1
    return boost_num_pos,boost_num_neg
```

接下来，你对否定词重复此过程。参见代码清单 3-12。

***代码清单 3-12.***

```
negation_words = set(["no","dont","didnt","cant","couldnt"])

def neg_chks(com_negation,pos_set1,neg_set1,full_txt_str):
    neg_num_pos = 0
    neg_num_neg = 0
    if(len(pos_set1)>0):
        for a in list(com_negation):
            for b in list(pos_set1):
                wrd_fnd = a + " " + b
                #print (wrd_fnd)
                if(full_txt_str.find(wrd_fnd)>=0):
                    #print (wrd_fnd,full_txt_str,"pos")
                    neg_num_pos = neg_num_pos +1
    if(len(neg_set1)>0):
        for a in list(com_negation):
            for b in list(neg_set1):
                wrd_fnd = a + " " + b
                if(full_txt_str.find(wrd_fnd)>=0):
                    #print (wrd_fnd,full_txt_str,"neg")
                    neg_num_neg = neg_num_neg +1
    return neg_num_pos,neg_num_neg
```



让我们做一个小测试来演示该功能。你将匹配的否定词与句子、匹配的肯定词集、匹配的否定词集（本例中为空集）以及句子本身传入。你会看到，你得到了一个肯定词被否定的命中结果（`“didn’t like”`）。参见代码清单 3-13。

### 代码清单 3-13

```
str_test ="it was given as a gift and the receiver didnt like it i wished
i had bought some other kind i was believing that ghirardelli would be the
best you could buy but not when it comes to peppermint bark this is not
their best effort."

neg_chks({"didnt"},{"like", "best"},{},str_test)

(1, 0)
```

## 感叹号（观察点 4）

接下来，我们讨论感叹号。感叹号会增强潜在情感（无论是正面还是负面）的强度。你需要检查句子中是否包含感叹号，并且该句子是否同时有正面或负面的命中结果。

你需要在句子级别进行检查，而不是在评论级别。与其它情况一样，你需要对 `Summary` 和 `full_txt` 都执行此操作。参见代码清单 3-14。

### 代码清单 3-14. 检查感叹号

```
def excl(pos_set1,neg_set1,full_txt_str):
    excl_pos_num=0
    excl_neg_num=0
    tok_sent = sent_tokenize(full_txt_str)
    for i in tok_sent:
        if(i.find('!')>=0):
            com_set = set(i.split()) & pos_set1
            if(len(com_set)>0):
                excl_pos_num= excl_pos_num+1
            else:
                com_set1 =set(i.split()) & neg_set1
                if(len(com_set1)>0):
                    excl_neg_num= excl_neg_num+1
    return excl_pos_num,excl_neg_num
```

## 对 `full_txt` 和 `Summary` 的评估（观察点 1）

代码清单 3-15 展示了首先获取匹配的正面和负面词汇集合的代码。这部分与你之前看到的类似。然后，它调用函数来获取 `full_txt` 和 `Summary` 列的强化词、否定词和感叹词的数量。接着，代码基于图 3-5 和图 3-6 中讨论的公式进行求和，并遵循图 3-7 中的逻辑来确定评论的最终标签，从而得到 `final_tag`。

代码清单 3-15 中有大量的列表初始化操作。你需要记录所有导致 `full_txt` 和 `Summary` 列总体得分的值。你需要这一步来后续分析结果并进一步优化结果。

### 代码清单 3-15

```
final_tag_list = []
pos_score_list = []
neg_score_list = []
pos_set_list = []
neg_set_list = []
neg_num_pos_list = []
neg_num_neg_list = []
boost_num_pos_list = []
boost_num_neg_list = []
excl_num_pos_list =[]
excl_num_neg_list =[]
pos_score_list_sum = []
neg_score_list_sum = []
pos_set_list_sum = []
neg_set_list_sum = []
neg_num_pos_list_sum = []
neg_num_neg_list_sum = []
boost_num_pos_list_sum = []
boost_num_neg_list_sum = []
excl_num_pos_list_sum =[]
excl_num_neg_list_sum =[]
```

在代码清单 3-16 中，`t3` 数据框的每一行都根据到目前为止讨论的步骤进行评估。

### 代码清单 3-16. 评估 `t3` 数据框

```
for i,row in t3.iterrows():
    full_txt_str = row["full_txt"]
    full_txt_set = set(full_txt_str.split())
    sum_txt_str = row["Summary"].lower()
    summary_txt_set = set(sum_txt_str.split())
    sent_len = len(full_txt_set)

    ####正面和负面集合
    pos_set1 = (full_txt_set) & (pos_set)
    neg_set1 = (full_txt_set) & (neg_set)
    com_pos_sum = (summary_txt_set) & (pos_set)
    com_neg_sum = (summary_txt_set) & (neg_set)

    ####强化词和否定词集合
    com_boost = (full_txt_set) & (booster_words)
    com_negation = (full_txt_set) & (negation_words)
    com_boost_sum = (summary_txt_set) & (booster_words)
    com_negation_sum = (summary_txt_set) & (negation_words)

    boost_num_pos=0
    boost_num_neg=0
    neg_num_pos=0
    neg_num_neg =0
    excl_pos_num=0
    excl_neg_num = 0
    boost_num_pos_sum=0
    boost_num_neg_sum=0
    neg_num_pos_sum=0
    neg_num_neg_sum =0
    excl_pos_num_sum=0
    excl_neg_num_sum = 0

    ####获取强化词、否定词和感叹词集合的计数器
```



```python
if(len(com_boost)>0):
    boost_num_pos, boost_num_neg = booster_chks(com_boost, pos_set1, neg_set1, full_txt_str)
    boost_num_pos_sum, boost_num_neg_sum = booster_chks(com_boost_sum, com_pos_sum, com_neg_sum, sum_txt_str)

if(len(com_negation)>0):
    neg_num_pos, neg_num_neg = neg_chks(com_negation, pos_set1, neg_set1, full_txt_str)
    neg_num_pos_sum, neg_num_neg_sum = neg_chks(com_negation_sum, com_pos_sum, com_neg_sum, sum_txt_str)

if((full_txt_str.find("!")>=0) & ((neg_num_pos+neg_num_neg)==0)):
    excl_pos_num, excl_neg_num = excl(pos_set1, neg_set1, full_txt_str)
    excl_pos_num_sum, excl_neg_num_sum = excl(com_pos_sum, com_neg_sum, sum_txt_str)
```

