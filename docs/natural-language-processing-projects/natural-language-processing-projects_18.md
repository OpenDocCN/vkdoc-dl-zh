# 从互联网文本生成随机新闻摘要

`model.predict('summarize: As many as 32 died following a landslide in Taliye village in Mahad Taluka, Raigad on Thursday. Around 35 houses were buried under debris and several people were still feared to be missing or trapped under it. In another landslide, four people died in Poladpur, which is also a landslide-prone area.')`

`['32 dead in landslide in Raigad village; 35 houses buried under debris']`

## 检查测试数据的标题

```
for doc in test_df['source_text']:
    print(model.predict(doc))
    print()
```

`['Punjab minister upgraded to Z plus, gets bullet-proof Land Cruiser']`

`summarize: Punjab minister Navjot Singh Sidhu's security cover has been upgraded to Z plus and a bullet-proof Land Cruiser from CM Captain Amarinder Singh's fleet has been given to him. Punjab government also asked Centre to provide Sidhu with Central Armed Police Forces cover, claiming that the "threat perception" to Sidhu increased after he attended Pakistan PM Imran Khan's swearing-in ceremony.`

`['Asthana appointed chief of Bureau of Civil Aviation Security']`

`summarize: After the Appointments Committee of the Cabinet curtailed his tenure as the CBI Special Director, Rakesh Asthana has been appointed as the chief of Bureau of Civil Aviation Security. Asthana and three other CBI officers' tenure was cut short with immediate effect. Last week, CBI Director Alok Verma was moved to fire services department, following which Verma announced his resignation.`

`['Marriage of Muslim man with Hindu woman merely irregular marriage: SC']`

`summarize: Citing Muslim law, the Supreme Court has said the marriage of a Muslim man with a Hindu woman "is neither a valid nor a void marriage, but is merely an irregular marriage". "Any child born out of such wedlock is entitled to claim a share in his father's property," the bench said. The court was hearing a property dispute case.`

`["I'm uncomfortable doing love-making scenes: Amrita Rao"]`

`summarize: Amrita Rao has said she's uncomfortable doing love-making scenes, while adding, "Love-making is so personal to me...if I do it on screen, it's like I'm leaving a part of my soul. I cannot do that," she added, "I'm not saying it's wrong...it's the reflection of how our society has changed." "It's just about a choice...we all make," Amrita further said.`

`['11-hr bandh across northeastern states on Jan 8 against Citizenship bill']`

`summarize: Student organisations and indigenous groups have called for an 11-hour bandh across the northeastern states on January 8 from 5 am-4 pm in protest against Centre's decision to table Citizenship (Amendment) Bill, 2016 in Parliament. The bill seeks to grant citizenship to minority communities, namely, Hindus, Sikhs, Christians, Parsis, Buddhists and Jains from Bangladesh, Pakistan and Afghanistan.`

`['Ex-Gujarat CM joins Nationalist Congress Party']`

`summarize: Former Gujarat Chief Minister Shankersinh Vaghela joined the Nationalist Congress Party (NCP) on Tuesday in presence of party chief Sharad Pawar. The 78-year-old who left Congress in 2017, on Friday said, "In public life, a good platform is required to raise issues." "Vaghela...is a dynamic leader who knows the pulse of...the country," Gujarat NCP chief Jayant Patel had said.`

结果看起来很有希望。我们可以进一步改进和优化摘要，以处理输出中的噪声。首先，让我们评估一下摘要模型。

## 摘要的评估指标

- **ROUGE 分数**：基本上，是指人工生成的摘要中出现在机器提取摘要中的单词数量。它从机器学习评估的角度衡量了输出的召回率。

- **BLEU 分数**：该分数使用以下公式计算。

`Score = number of words in the machine-generated summary that appeared in the human-created summary.`

BLEU 分数给出了模型的精确度。

让我们快速计算一下 T5 模型的 BLEU 分数。

让我们导入库。

```
from nltk.translate.bleu_score import sentence_bleu
```

让我们对新的源文本进行预测。

```
x = [x for x in df.headlines]
y = [model.predict(p)[0] for p in df['source_text']]
# Function to calculate the score
L = 0
for i, j in zip(x, y):
    L += sentence_bleu(
        [i],
        j,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=None,
        auto_reweigh=False,
    )
# Average blue score of whole corpuses
L / df.shape[0]
0.769163055697453
```

整个语料库的平均分数为 77%。

## 未来展望

输出必须语法正确、可读性强且格式良好。为了在生成摘要后也能实现这一点，我们需要进行输出处理。这种后处理是在单词或句子级别进行的，以使输出易于理解。

以下是我们可以在流程中添加的一些后处理步骤。

1. **检查语法**。鉴于输出是机器生成的，我们可以预期会有很多语法错误。在向用户展示之前，我们需要清理它们。

   我们可以使用像 Grammarly 的 GECToR（语法错误纠正：标记，而非重写）这样的库（`https://github.com/grammarly/gector`）。

2. **删除重复内容**。摘要中的另一个最大挑战是重复——单词和句子的重复。我们也需要删除它们。

3. **删除不完整的句子**。我们需要过滤掉模型生成的不完整句子（如果有的话）。

## 结论

随着来自各类媒体和社交媒体的内容和新闻不断增加，文本摘要是 NLP 和深度学习应用最广泛的领域之一。我们探讨了如何构建从导入原始数据、清洗数据、使用预训练模型进行自定义训练、评估，到最终预测（生成摘要）的完整流程。在添加更多数据、更改参数以及使用一些高端 GPU 进行更好训练方面，还有很大的空间，这将产生出色的结果。

应用在各个行业都非常广泛，我们已经可以通过该领域正在进行的几个研究项目看到其发展势头。强化学习与深度学习的结合有望带来更好的结果，相关研究也在朝着这个方向继续推进。目前已有许多用于文本摘要的 API，但它们还不够准确和可靠。随着所有这些进步，文本摘要很快将达到一个全新的水平。

