# 第 3 章 在线评论中的自然语言处理

### 情感分析库

也有一些标准库可用于进行情感分析。它们提供开箱即用的情感分析，因此必须适当使用。如果它适合手头的问题，那么你就有了一个快速且稳健的解决方案。你也可以将其与他们使用的算法/方法结合使用，或作为独立解决方案使用。`Vader`（情感感知词典和情感推理器）是一个进行情感分析并提供带有权重的词极性的`Python`库。原始研究论文题为“Vader: A parsimonious rule-based model for sentiment analysis of social media text”，可在[`www.aaai.org/ocs/index.php/ICWSM/ICWSM14/paper/viewPaper/8109`](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM14/paper/viewPaper/8109)找到。从根本上说，`Vader`从成熟的词典以及社交媒体语料库中获取了一组词汇。每个词都由一组人员按照群体智慧的方法提供极性分数，以得出最终的词典及其权重。清单 3-27 展示了如何使用`pip install`安装`Vader`库。



***列表 3-27.*** 安装 Vader 库

```python
!pip install vaderSentiment
```

列表 3-28 提供了一种使用 Vader 库的快速方法。

***列表 3-28.*** 使用 Vader 库

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

score = analyser.polarity_scores("I am good")

score
```

`{'neg': 0.0, 'neu': 0.256, 'pos': 0.744, 'compound': 0.4404}`

此示例中的复合分数是该句子情感极性的最终加权分数。正面和负面分数是处于该极性的词汇所占的百分比。它们的总和为 1。

第 3 章 在线评论中的 NLP

## 方法 3：基于机器学习的方法（神经网络）

由于您拥有用户提供的最终评分，您可以尝试拟合一个监督模型。在您的情况下，进行监督模型的一种方法是使用 `Summary` 和 `Text` 列中的所有文本作为特征，`score_bkt` 作为目标变量，并拟合一个机器学习模型。这种方法的一个问题是，模型可能会因产品名称或评分较高或较低的产品类别而产生偏差。另一种方法是使用您之前创建的词汇特征作为模型中的特征。表 3-5 显示了您在基于词汇的方法中创建的中间变量。您将在此基础上添加更多变量来构建您的模型。

***表 3-5.** 中间变量*

| 词汇特征 | 变量含义 |
| --- | --- |
| `sent_len` | 文本长度 |
| `pos_score` | 词汇方法得出的正面分数 |
| `neg_score` | 词汇方法得出的负面分数 |
| `neg_num_pos_count` | 正面词汇附近的负面词汇数量 |
| `neg_num_neg_count` | 正面词汇附近的负面词汇数量 |
| `boost_num_pos_count` | 正面词汇附近的强化词数量 |
| `boost_num_neg_count` | 负面词汇附近的强化词数量 |
| `neg_num_neg_count_sum` | 摘要中正面词汇附近的负面词汇数量 |
| `boost_num_pos_count_sum` | 摘要中正面词汇附近的负面词汇数量 |
| `boost_num_neg_count_sum` | 摘要中正面词汇附近的强化词数量 |
| `excl_num_pos_count` | 正面词汇附近的感叹号数量 |
| `excl_num_neg_count` | 负面词汇附近的感叹号数量 |
| `excl_num_pos_count_sum` | 摘要中正面词汇附近的感叹号数量 |
| `excl_num_neg_count_sum` | 摘要中负面词汇附近的感叹号数量 |

第 3 章 在线评论中的 NLP

## 语料库特征

利用语料库的文本和摘要，您可以使用 TF-IDF 向量化器创建词袋特征。由于您想创建一个通用的情感分析模块，您只从语料库中选择与情感或情绪相关的词汇。这是一个重要的步骤，因为使用所有词汇（在您的情况下，如宠物食品名称、糖果名称等）可能会使模型学习到将产品与其相关情感关联起来的模式。您将创建两组词汇特征：仅包含形容词和仅包含停用词。请注意，在第 2 章的分类示例中，您剔除了停用词。在这里，您将仅使用停用词作为特征集。

然后，您将合并这两组特征，并以 `score_bkt` 作为因变量来训练一个神经网络，调整类别权重，以得到一个能够在所有级别上进行良好分类的最佳模型。您在此处使用的是 NLTK 3.4.3 版本，如表 3-7 所述。请参见列表 3-29 和列表 3-30。

***列表 3-29.***

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
import nltk
import warnings
import stop_words
warnings.filterwarnings('ignore')
```

***列表 3-30.***

```python
t1 = pd.read_csv("lexicon_sent_processed.csv")
tgt = t1.loc[:,"score_bkt"]
```

以下代码展示了获取文本特征的函数。函数 `cnv_str`



