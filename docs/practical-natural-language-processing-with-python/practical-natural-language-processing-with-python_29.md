# 第 3 章 在线评论中的自然语言处理

在`t1`数据集中有两列文本：`Text`和`Summary`。你将合并它们并处理成一个单独的列。正是这个列将进行词典挖掘。参见清单 3-3。

**清单 3-3.** 合并列

```python
t1["full_txt"] = t1["Summary"] + " " + t1["Text"]
t1["full_txt"] = t1["full_txt"].str.lower()
t1["sent_len"] = t1["full_txt"].str.count(" ") + 1
```

清单 3-4 帮助你排除包含缺失单词的句子。

**清单 3-4.** 优化数据集

```python
t2 = t1[t1.sent_len>=1]
len(t1),len(t2)
(568454, 568427)
```

为了衡量你的方法的准确性，你将把客户在`Score`列中提供的最终评分分桶。参见清单 3-5。

**清单 3-5.**

```python
##for meausring accuracy
t2["score_bkt"]="neu"
t2.loc[t2.Score>=4,"score_bkt"] = "pos"
t2.loc[t2.Score<=2,"score_bkt"] = "neg"
```

最简单的方法是遍历句子语料库中的所有单词，并与词典列表进行匹配。由于这些是较长的句子，你需要通过句子中的单词数量来标准化正面和负面命中的数量。之后，对正面、负面和中性分数进行简单比较，并根据哪个分数更高来标记句子。为了进一步改进并提出额外策略，你还将记录句子中出现的单词列表，作为`pos_set_list`和`neg_set_list`。参见清单 3-6。

**清单 3-6.**

```python
final_tag_list = []
pos_percent_list = []
neg_percent_list = []
pos_set_list = []
neg_set_list = []
for i,row in t3.iterrows():
    full_txt_set = set(row["full_txt"].split())
    sent_len = len(full_txt_set)
    pos_set1 = (full_txt_set) & (pos_set)
    neg_set1 = (full_txt_set) & (neg_set)
    com_pos = len(pos_set1)
    com_neg = len(neg_set1)
    if(com_pos>0):
        pos_percent = com_pos/sent_len
    else:
        pos_percent = 0
    if(com_neg>0):
        neg_percent = com_neg/sent_len
    else:
        neg_percent =0
    if(pos_percent>0)|(neg_percent>0):
        if(pos_percent>neg_percent):
            final_tag = "pos"
        else:
            final_tag = "neg"
    else:
        final_tag="neu"
    final_tag_list.append(final_tag)
    pos_percent_list.append(pos_percent)
    neg_percent_list.append(neg_percent)
    pos_set_list.append(pos_set1)
    neg_set_list.append(neg_set1)
```

将创建的列表赋值给`t3`数据框的操作在清单 3-7 中完成。

**清单 3-7.** 赋值列表

```python
t3["final_tags"] = final_tag_list
t3["pos_percent"] = pos_percent_list
t3["neg_percent"] = neg_percent_list
t3["pos_set"] = pos_set_list
t3["neg_set"] = neg_set_list
```

现在你已经将句子分类为正面、负面和中性三个类别。你需要检查它们是否与之前讨论的`score_bkt`一致。你将使用准确率分数、F1 分数和混淆矩阵来衡量。更多关于这些指标的详细信息，可以从文章《Accuracy, Precision, Recall & F1 Score: Interpretation of Performance Measures》（[`blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/`](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)）中学习。



