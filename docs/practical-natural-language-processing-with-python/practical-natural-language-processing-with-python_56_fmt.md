# 第 5 章：虚拟助手中的自然语言处理

接下来，你需要替换特殊字符，并使用句子分隔符 `sep_sent` 来分割机器人语句。这将帮助你模板化每条机器人语句，并随后从解码器中生成这些模板。请参见 `代码清单 5-29`。

```python
t4["corrected_bot1"] = t4["corrected_bot"].str.replace('\xa0','')
t4["corrected_bot1"] = t4["corrected_bot1"].str.replace('[^a-z0-9\s\.\?]+','')
t4["corrected_bot1"] = t4["corrected_bot1"].str.replace('[0-9]{1,3}',' short_num ')
t4["corrected_bot1"] = t4["corrected_bot1"].str.replace('[0-9]{4,}',' long_num ')
t4["corrected_bot1"] = t4["corrected_bot1"].str.replace('(\.\s){1,3}',' sep_sent ')
t4["corrected_bot1"] = t4["corrected_bot1"].str.replace('(\.){1,3}',' sep_sent ')
t4["corrected_bot1"] = t4["corrected_bot1"].str.replace('\?',' sep_sent ')
t4["corrected_bot1"] = t4["corrected_bot1"].str.replace('(sep_sent )+',' sep_sent ')
```

## 替换客户语句中的特殊字符

替换客户语句中的特殊字符是为了降低数据的稀疏性。例如，考虑两个句子："Is this correct..." 和 "Is this correct"。在这两种情况下，单词 "correct" 是相同的，只是前者有三个点，后者没有。在进行分词时，你需要确保这两个单词被同等对待。请参见 `代码清单 5-30`。

```python
t4["corrected_cust"] = t4["corrected_cust"].str.replace('(\.\s){1,3}',' ')
t4["corrected_cust"] = t4["corrected_cust"].str.replace('(\.){1,3}',' ')
```

## 将机器人语句转换为模板句子

由于你的语料库是机器人与客户之间的交互，一些句子在不同对话中是重复的。考虑到客服数据的性质，即使是人与人之间的对话，不同对话中也会出现大量重复的句子。这些重复的句子可以转换为模板。例如，在询问公交车到达时间时，表达公交车到达时间的方式只有有限的几种。这可以极大地帮助你规范化数据并生成有意义的输出。在 `代码清单 5-31` 中，你将机器人语句转换为有意义的模板句子。这些模板将被映射到模板编号。请参见 `图 5-16`。

```python
templ = pd.DataFrame(t4["corrected_bot1"].value_counts()).reset_index()
templ.columns = ["sents","count"]
templ.head()
```

`图 5-16`

## 为句子分配模板 ID

你将所有唯一的句子放入一个列表中。接下来，你重新构建列表为一个数据框，以便为每个句子分配一个模板 ID。请参见 `代码清单 5-32` 和 `图 5-17`。

```python
df = templ.sents.str.split("sep_sent",expand=True)
sents_all = []
for i in df.columns:
    l1 = list(df[i].unique())
    sents_all = sents_all + l1
sents_all1 = set(sents_all)
sents_all_df = pd.DataFrame(sents_all1)
sents_all_df.columns = ["sent"]
sents_all_df["ind"] = sents_all_df.index
sents_all_df["ind"] = "templ_" + sents_all_df["ind"].astype('str')
sents_all_df.head()
```

`图 5-17`

## 将句子分配给模板 ID

现在，你获取原始数据集 `t4`，并将不同的句子分配给模板 ID。首先，你创建一个字典，以句子为键，模板 ID 为值。请参见 `代码清单 5-33` 和 `5-34`。

```python
sent_all_df1 = sents_all_df.loc[:,["sent","ind"]]
df_sents = sent_all_df1.set_index('sent').T.to_dict('list')
```

```python
t4["corrected_bots_sents"] = t4["corrected_bot1"].str.split("sep_sent")
index_list=[]
for i, row in t4.iterrows():
    sent_list = row["corrected_bots_sents"]
    str_index = ""
    for j in sent_list:
        if(len(j)>=3):
            str_index = str_index + " " + str(df_sents[j][0])
    index_list.append(str_index)
t4["bots_templ_list"] = index_list
```

## 准备训练数据

你的训练集是机器人提问、客户回答的形式。但为了训练，你的输入将是客户句子，输出是机器人的响应。因此，你将 `corrected_bot1` 向后移动一个句子。这样，语料库就变成了输入的客户文本和正确的机器人输出。你移动 `user_id` 以标记聊天的结束。请参见 `代码清单 5-35` 和 `图 5-18`。

```python
t4["u_id_shift"] = t4["user_id"].shift(-1)
t4["corrected_bot1_shift"] = t4["corrected_bot1"].shift(-1)
t4["bots_templ_list_shift"] = t4["bots_templ_list"].shift(-1)
t4.loc[t4.u_id_shift!=t3.user_id,"corrected_bot1_shift"] = "end_of_chat"
t4.loc[t4.u_id_shift!=t3.user_id,"bots_templ_list_shift"] = "end_of_chat"
req_cols = ["user_id","corrected_cust","corrected_bot1_shift","bots_templ_list_shift"]
t5 = t4[req_cols]
t5.tail()
```

`图 5-18`

## 保存模板映射字典

现在，你选取字典 `df_sents` 的输出，用于将模板映射回原始句子。在编码器中，你使用模板 ID 作为输出。然后，你使用字典重新映射，以打印实际的句子。请参见 `代码清单 5-36`。

```python
import pickle
output = open('dict_templ.pkl', 'wb')
```