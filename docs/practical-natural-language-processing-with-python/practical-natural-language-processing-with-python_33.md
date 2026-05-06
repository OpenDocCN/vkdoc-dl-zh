# 第 3 章 在线评论中的自然语言处理

#### 计算总体得分

```python
score_pos = len(pos_set1) + boost_num_pos - 2*neg_num_pos + 2*excl_pos_num
score_pos_sum = len(com_pos_sum) + boost_num_pos_sum - 2*neg_num_pos_sum + 2*excl_pos_num_sum
score_pos = score_pos + 1.5*score_pos_sum

score_neg = len(neg_set1) + 3*len(com_neg_sum) + boost_num_neg - 2*neg_num_neg + 2*excl_neg_num
score_neg_sum = len(com_neg_sum) + boost_num_neg_sum - 2*neg_num_neg_sum + 2*excl_neg_num_sum
score_neg = score_neg + 1.5*score_neg_sum
```

#### 最终判定

```python
if((score_pos>0)|(score_neg>0)):
    if((score_neg>score_pos)):
        final_tag = "neg"
    elif(score_pos>score_neg):
        final_tag = "pos"
    else:
        if(full_txt_str.find("!")>=0):
            final_tag = "pos"
        else:
            final_tag = "neu"
else:
    if(full_txt_str.find("!")>=0):
        final_tag="pos"
    else:
        final_tag="neu"
```

#### 记录中间值以便排查问题

```python
final_tag_list.append(final_tag)
pos_score_list.append(score_pos)
neg_score_list.append(score_neg)
pos_set_list.append(pos_set1)
neg_set_list.append(neg_set1)
neg_num_pos_list.append(neg_num_pos)
neg_num_neg_list.append(neg_num_neg)
boost_num_pos_list.append(boost_num_pos)
boost_num_neg_list.append(boost_num_neg)
excl_num_pos_list.append(excl_pos_num)
excl_num_neg_list.append(excl_neg_num)
pos_set_list_sum.append(com_pos_sum)
neg_set_list_sum.append(com_neg_sum)
neg_num_pos_list_sum.append(neg_num_pos_sum)
neg_num_neg_list_sum.append(neg_num_neg_sum)
boost_num_pos_list_sum.append(boost_num_pos_sum)
boost_num_neg_list_sum.append(boost_num_neg_sum)
excl_num_pos_list_sum.append(excl_pos_num_sum)
excl_num_neg_list_sum.append(excl_neg_num_sum)
```

清单 3-17 将所有值设置到数据框中，以便理解准确率并识别任何改进机会。

**清单 3-17.**

```python
t3["final_tags"] = final_tag_list
t3["pos_score"] = pos_score_list
t3["neg_score"] = neg_score_list
t3["pos_set"] = pos_set_list
t3["neg_set"] = neg_set_list
t3["neg_num_pos_count"] = neg_num_pos_list
t3["neg_num_neg_count"] = neg_num_neg_list
t3["boost_num_pos_count"] = boost_num_pos_list
t3["boost_num_neg_count"] = boost_num_neg_list
t3["pos_set_sum"] = pos_set_list_sum
t3["neg_set_sum"] = neg_set_list_sum
t3["neg_num_pos_count_sum"] = neg_num_pos_list_sum
t3["neg_num_neg_count_sum"] = neg_num_neg_list_sum
t3["boost_num_pos_count_sum"] = boost_num_pos_list_sum
t3["boost_num_neg_count_sum"] = boost_num_neg_list_sum
t3["excl_num_pos_count"] = excl_num_pos_list
t3["excl_num_neg_count"] = excl_num_neg_list
t3["excl_num_pos_count_sum"] = excl_num_pos_list_sum
t3["excl_num_neg_count_sum"] = excl_num_neg_list_sum
```

在清单 3-18 中，你计算准确率、混淆矩阵和 F1 分数。

**清单 3-18.** 准确率、混淆矩阵和 F1 分数

```python
from sklearn.metrics import accuracy_score
print (accuracy_score(t3["score_bkt"],t3["final_tags"]))
# 0.8003448093872596

from sklearn.metrics import f1_score
f1_score(t3["score_bkt"],t3["final_tags"], average='macro')
# 0.502500231042986
```

你可以看到清单 3-18 中的数值有所改善。接下来，让我们查看清单 3-19 和图 3-8 中的混淆矩阵。

**清单 3-19.** 混淆矩阵

```python
rows_name = t3["score_bkt"].unique()
from sklearn.metrics import confusion_matrix
```



```python
cmat = pd.DataFrame(confusion_matrix(t3["score_bkt"],t3["final_tags"], labels=rows_name, sample_weight=None))
cmat.columns = rows_name
cmat["act"] = rows_name
cmat
```

![图片 27](img/index-97_1.png)

