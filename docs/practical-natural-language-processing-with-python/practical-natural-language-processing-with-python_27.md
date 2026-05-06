# 目录

- 历史
- 基于规则与统计的 NLP
- 主要评估与任务
- 句法、语义、语篇、语音、对话
- 参见、参考文献、扩展阅读

## 历史

自然语言处理（NLP）的历史通常始于 20 世纪 50 年代，尽管更早时期也有相关研究。1950 年，艾伦·图灵发表了一篇题为《计算机器与智能》的文章。

下一步是从句子中获取二元词组语料库。在构建语言模型时，一旦你提供某个单词，模型就会输出该单词之后最可能出现的单词。你可以将 n-gram 的阶数提高以获得更好的精度。参见列表 2-42。

**列表 2-42.**

```
n_grams = ngrams(inp_text1.split(), 2)
l1 = []
for grams in n_grams :
    l1.append((grams[0].lower(),grams[1].lower()))
```

## 第 2 章：NLP 在客户服务中的应用

现在，二元词组被组织成一个包含前导词和滞后词的数据框。因此，给定一个前导词，你可以得到一组可能的滞后词。参见列表 2-43 和 2-44。

**列表 2-43.**

```
df0 = pd.DataFrame(l1)
df0.columns = ["lead","lag"]
lead_all = df0["lead"].unique()
lag_all = df0["lag"].unique()
```

**列表 2-44.**

```
lead_dict = {}
for i in lead_all:
    matches = df0.loc[df0.lead==i,"lag"]
    len_mtch = len(matches)
    lag_dict = dict(Counter(matches))
    for k in lag_dict:
        lag_dict[k] = lag_dict[k]/len_mtch
    lead_dict[i] = lag_dict
```

让我们用单词`"language"`来测试一下。列表 2-45 展示了单词`"language"`之后最可能出现的单词。

**列表 2-45.** 最可能单词测试

```
lead_dict["language"]
```

输出：

```
{'processing': 0.6666666666666666,
 'data': 0.041666666666666664,
 'understanding': 0.041666666666666664,
 'generation': 0.041666666666666664,
 'system': 0.041666666666666664,
 'models': 0.041666666666666664,
 'tasks': 0.041666666666666664,
 'modeling': 0.08333333333333333}
```

这个语言模型也用于其他某些用例，比如机器翻译或自然语言生成。回到你的例子，声学模型的输出会通过语言模型进行校正，然后得到语音到文本的输出。一旦你得到文本输出，就可以对其进行挖掘，并得出所有见解。

到目前为止，你已经看到了 NLP 在客户服务行业中的大量用例。还有一些更细微的用例，比如提取用户画像、随着对话进行情感的变化、不同客服渠道（如语音和文本）的关联等等。这里涵盖的用例应该足以让任何涉足该领域的人入门。客户服务领域拥有丰富的文本数据，分析这些数据的不同维度可以为组织带来显著影响。

## 第 3 章：NLP 在在线评论中的应用

如今，评论是在线购买周期的重要组成部分。尽管“发声的少数人”数量不多，但受评论影响的用户数量却非常庞大。一项研究发现，63%的用户更喜欢有评论的在线网站。访问评论页面的顾客从网站购买的可能性高出惊人的 105%（[`cxl.com/blog/user-generated-reviews/#:~:text=Reevoo%20found%20that%2050%20or,site%20that%20has%20user%20reviews`](https://cxl.com/blog/user-generated-reviews/#:~:text=Reevoo%20found%20that%2050%20or,site%20that%20has%20user%20reviews)）。挖掘这些评论可以为在线服务提供商以及上架产品的卖家提供见解。除了了解顾客是否满意之外，我们还可以知道用户对



