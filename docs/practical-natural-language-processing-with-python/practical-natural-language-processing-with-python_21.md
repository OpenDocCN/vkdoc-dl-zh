# 第 2 章 客户服务中的自然语言处理

### 代码清单 2-15

```
###获取唯一词列表

list_all = []

for i in df["token_words"]:

list_all = list_all + i

list_words = list(set(list_all))

list_words[0:5]

['check', 'do', '20th', 'hello', 'monday']

###初始化数组：行数=数据框长度，列数=唯一词数量

arr1 = np.zeros([len(df),len(list_words)])

###遍历数据框，将句子中出现的词标记为 1

for i,row in df.iterrows():

token_words = row["token_words"]

for j in token_words:

if (j in list_words):

k = list_words.index(j)

arr1[i][k] = 1

###验证结果是否正确

get_non_zero = np.where(arr1[0]==1)

list_index = list(get_non_zero[0])

[list_words[i] for i in list_index]

['can', 'when', 'web', 'i']
```

### 图 2-8 词项-文档矩阵

