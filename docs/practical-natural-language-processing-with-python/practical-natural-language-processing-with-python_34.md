# 第 3 章 在线评论中的自然语言处理

**图 3-8.** 混淆矩阵

如您所见，负面和正面分类的效果远好于中性分类。这拉低了您的整体 F1 分数（宏平均）。然而，在中性分类上，您的表现相当糟糕。您也可以通过从数据集中移除中性分类来重新检查分数。请参见代码清单 3-20。

**代码清单 3-20.** 从数据集中移除中性分类

```python
t4 = t3.loc[(t3.score_bkt!="neu") & (t3.final_tags!="neu")].reset_index()
print(accuracy_score(t4["score_bkt"],t4["final_tags"]))
print(f1_score(t4["score_bkt"],t4["final_tags"],average='macro'))
0.8812103027705257
0.75425508403348
```

## 优化代码

显然，您的策略在正面和负面分类上效果良好。因此，在下一轮迭代中，您应重点关注并优化中性情感的分类。以下是下一轮优化的一些策略（既针对中性分类，也适用于其他分类）。您可以继续推进这个问题，运用任何您能想到的策略来解决它，并将当前的分数作为基准。

- **大小写**：大写单词表示强烈的情感，类似于感叹号。有时大写单词本身就是情感词，或者它们可能紧邻情感词，例如“EXTREMELY happy”或“EXTREMELY angry with the product”。您需要相应地识别大写单词并调整分数。

![图片 28](img/index-98_1.png)

![图片 29](img/index-98_2.png)

- **情感词典**：您可以尝试使用不同的情感词典。还有一些词典为不同的情感词提供了权重。论文《基于词典的极端观点搜索方法》（[`journals.plos.org/plosone/article?id=10.1371/journal.pone.0197816`](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0197816)）描述了一种从现有在线评分中生成词典的方法。其思路是利用给定评论中带有评分的词语作为词典来源，然后对所有评论中的这些评分进行归一化处理，从而得到一个带有权重标度的词典。该词典随后可用于任何分类目的。

- **使用语义倾向性确定极性**：另一种方法是利用网络或任何其他外部来源来确定当前词语的极性。例如，您可以查看单词“delight”与单词“good”共同出现的页面点击量。您还可以查看单词“delight”与单词“bad”共同出现的页面点击量。通过比较结果，您可以判断单词“delight”应具有正面还是负面极性。以下方法来自论文《赞还是踩？：语义倾向性在无监督评论分类中的应用》（[`dl.acm.org/doi/10.3115/1073083.1073153`](https://dl.acm.org/doi/10.3115/1073083.1073153)）。已知的正面词（例如“good”）与给定词典之间的相似度是基于点互信息以及杰卡德指数来衡量的。公式如图 3-9 和图 3-10 所示。

**图 3-9.**

**图 3-10.**

这里 `P(Word1)` 或 `P(Word2)` 指的是该词在网络搜索引擎中的点击量。`P(Word1 & Word2)` 指的是在同一查询中，单词 1 和单词 2 共同出现的点击量。代码清单 3-21 和 3-22 演示了针对几个词语的方法。您可以通过网络搜索结果，获取给定词语与

![图片 30](img/index-99_1.jpg)

![图片 31](img/index-99_2.png)



一个感兴趣的短语。这可以扩展到词典中的所有词汇，也可以对语料库句子中的形容词进行类似处理。对于以下练习，你需要一个 Bing API 密钥来获取 Bing 搜索引擎中的命中次数。创建 Bing API 密钥的步骤如下：

1.  访问 [`azure.microsoft.com/en-us/try/cognitive-services/`](https://azure.microsoft.com/en-us/try/cognitive-services/)。
2.  在图 3-11. 所示的屏幕中，点击“搜索 API”选项卡。

***图 3-11.** 搜索 API*

3.  点击图 3-12 中所示的“获取 API 密钥（Bing 搜索 API V7）”按钮。

***图 3-12.** 获取 API 密钥*

![Image 32](img/index-100_1.png)

第 3 章 在线评论中的自然语言处理

4.  系统会要求你登录，之后你将看到一个类似于图 3-13 的页面。向下滚动以获取密钥 1。

***图 3-13.** 获取密钥 1*

一旦你获得了 Bing API，请按照以下步骤操作：

1.  获取锚词的命中次数。对于正面词，考虑单词“good”；对于负面词，考虑单词“bad”。
2.  对于感兴趣单词列表中的每个单词，获取其总命中次数。
3.  对于每个单词，获取该单词与每个锚词共同出现的总命中次数。
4.  计算 PMI 和 Jaccard 系数。当命中次数低于某个阈值时，将该分数视为无效。这是因为很大一部分命中次数可能纯属偶然。由于这是网络搜索，我们将阈值设定在数百万级别。此方法基于论文《使用网络搜索引擎测量词语间的语义相似度》，论文地址为 [`www2007.org/papers/paper632.pdf`](http://www2007.org/papers/paper632.pdf)。
5.  比较分数并标记情感极性。

第 3 章 在线评论中的自然语言处理

让我们开始吧。参见清单 3-21。

***清单 3-21.***

```
import requests
import numpy as np

search_url = "https://api.cognitive.microsoft.com/bing/v7.0/search"
headers = {"Ocp-Apim-Subscription-Key": "your-key"}
```

在清单 3-22 中，`get_total` 函数用于获取给定查询的总词数，`get_query_word` 函数通过将感兴趣单词与正面和负面锚词组合来准备查询。

***清单 3-22.***

```
def get_total(query_word):
    params = {"q": query_word, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    return search_results['webPages']['totalEstimatedMatches']

def get_query_word(str1, gword, bword):
    str_base = gword
    str_base1 = bword
    query_word_pos = str1 + "+" + str_base
    query_word_neg = str1 + "+" + str_base1
    return query_word_pos, query_word_neg
```

在清单 3-23 中，`get_pmi` 和 `get_jaccard` 函数根据图 3-10. 中描述的公式，为所有输入参数计算逐点互信息分数和 Jaccard 分数。如果命中次数不足，则将其标记为“na”。

第 3 章 在线评论中的自然语言处理

***清单 3-23.***

```
def get_pmi(hits_good, hits_bad, hits_total, sr_results_pos_int, sr_results_neg_int, str1_tot):
    pos_score = "na"
    neg_score = "na"
    if(sr_results_pos_int >= 1000000):
        pos_score = np.log((hits_total * sr_results_pos_int) / (hits_good * str1_tot))
    if(sr_results_neg_int >= 1000000):
        neg_score = np.log((hits_total * sr_results_neg_int) / (str1_tot * hits_bad))
    return pos_score, neg_score

def get_jaccard(hits_good, hits_bad, sr_results_pos_int, sr_results_neg_int, str1_tot):
    pos_score = "na"
    neg_score = "na"
    if(sr_results_pos_int >= 1000000):
        pos_score = sr_results_pos_int / (((str1_tot + hits_good) - sr_results_pos_int))
    if(sr_results_neg_int >= 1000000):
        neg_score = sr_results_neg_int / (((str1_tot + hits_bad) - sr_results_neg_int))
    return pos_score, neg_score
```

现在，你需要定义锚词以及你想要分析的感兴趣单词。



理解极性（`list1`）。对于锚点词，你需要理解当这些词中的任意一个出现时的总命中数（你将这个空间定义为全集）。这是计算`PMI`和`Jaccard`分数所必需的。另一点需要注意的是，由于你感兴趣的是评论中通常表达的情感词汇，因此你将单词“reviews”添加到搜索范围中。这会将上下文限定在情感领域。这些是我已经应用的优化。请根据手头的问题随意修改。参见清单 3-24。

