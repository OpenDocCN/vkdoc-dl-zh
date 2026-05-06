# 第 2 章 客户服务中的自然语言处理

产品可能存在多种变体。例如，冰箱可以被称为 fridge，也可以用品牌名如三星、型号名如 687L 来指代，有时还会用品牌加容量的方式如三星 200L。鉴于市场上产品种类繁多且引用方式多样，为零售或电商提取产品信息非常复杂。你将在第 3 章详细了解这一点。对于电信或银行、金融服务与保险等其他领域，产品变体不多，因此可以通过简单的查找来处理。图 2-18 展示了一个图表，你可以根据转化率和数量进行分析并获得洞察。它显示了转化率最高的产品。

**图 2-18.** 热门产品及转化率

## 未购买原因

你还可以挖掘聊天记录中客户对购买产品提出的异议。通常，一个针对异议训练的良好分类器可以很好地完成这项工作。在这里，定位问题所在行是关键。并非聊天中的每一行都包含客户的异议。有时，客服人员也会被提示向客户提出引导性问题，以让他们陈述异议。未购买的原因也会作为最终客服处理结果的一部分被收集。客户通常不购买的主要原因包括价格、产品不兼容、安装问题、退货政策、运费、竞争对手提供更优惠的交易，以及在做出购买决定前想咨询他人。这些高层次原因在不同公司之间是相似的，因此你可以为不同组织使用相同的预测模型（甚至启发式规则）。

## 调查评论分析

在聊天结束时收集调查问卷，以了解客户体验。这个数据集在试图理解客户对客服或产品的看法，甚至未购买原因方面非常丰富。会应用一组启发式规则或监督模型来挖掘调查评论。这个语料库也是用户情绪或情感的良好指标。聊天对话通常趋于中性，而调查评论可以清晰地表明客户的情感。你将在下一章详细探讨情感挖掘。

## 挖掘语音转录文本

到目前为止，你已经详细探讨了客户服务中的文本挖掘。然而，大部分数据仍然以通话录音的形式在呼叫中心产生。一旦你将语音转换为文本，就可以应用你的文本技术。你将使用来自 `https://pypi.org/project/SpeechRecognition/` 的语音识别包，并利用 Google API 将语音转换为文本。你可以使用清单 2-35 中的代码通过 `pip install` 安装该包。

**清单 2-35.** 安装包

```
!pip install SpeechRecognition
```

你以 `.wav` 文件作为输入，并使用 Google API 来获取转录的句子。参见清单 2-36。

**清单 2-36.** 转录句子

```
import speech_recognition as sr

r = sr.Recognizer()

def speech_to_text(af, show_all):
    #playsound(af)
    with sr.AudioFile(af) as source:
        #读取音频文件。这里我们使用 record 而不是
        #listen
        audio = r.record(source)
        abc = r.recognize_google(audio, show_all=show_all)
        return abc
```

该函数接受两个参数。一个是文件本身。另一个是语音识别的标志，名为 `show_all`。如果此标志为 `False`，则 Google API 返回最可能的语音转文本识别结果。输出如清单 2-37 所示。

**清单 2-37.**

```
af1 = "sample_audio.wav"
speech_to_text(af1, False)
'I am calling for reversal of enquiry and my account number is 911'
```

将 `show_all` 设置为 `False`，Google API 会返回所有识别结果，并附带



`confidence level`作为一个字典。如果你希望获取最可能的输出，这会很有用。现在请参阅清单 2-38。

***清单 2-38.***

`speech_to_text(af1,True)`

```
{'alternative': [{'transcript': 'I am calling for reversal of
enquiry and my account number is 911',
'confidence': 0.89367926},
{'transcript': 'I am calling for reversal of enquiry in my account
number is 911'},
{'transcript': 'I am calling for reversal of enquiry and my account
number is 9 11'},
{'transcript': "I'm calling for reversal of enquiry and my account
number is 911"},
{'transcript': 'I am calling for the reversal of enquiry and my account
number is 911'}],
'final': True}
```

该包还支持其他 API，如 Bing、IBM 和 Spinx。请访问
[`pypi.org/project/SpeechRecognition/.`](https://pypi.org/project/SpeechRecognition/)

你也可以从头开始使用神经网络模型构建自己的语音识别工具。这超出了本书的范围，但我将总结语音识别中包含的高级步骤。有两个基本部分：一个是将语音转换为文本的声学模型，另一个是使文本更有意义和连贯的语言模型。

## 声学模型

第一步是从音频文件中获取数据以及相应的标签。
有许多开源数据集可以帮助你开始这一步。获得文件后，你使用数字信号处理技术将`.wav`文件转换为数字。一些标准做法是使用 MFCC 或频谱图方法进行转换。

## 语言模型

完成此操作后，你会得到一个数字数组和相应的标签。然后，你可以训练一个神经网络模型，将给定的数组集合分类为它可能包含的单词。在评分新的音频文件时，首先将文件分割成更小的音频块，然后使用语音识别模型将每个块识别为一个单词。一旦你获得了识别的单词集，将它们转换为有意义且最可能的句子或短语就是语言模型的任务。在某些情况下，声学模型被训练用于检测单词的字母。语言模型则获取最可能的单词。

语言模型基于单词共同出现的概率。可以有一种纯基于频率的方法，也可以有一种基于循环神经网络的序列到序列模型，用于纠正从声学模型输出的句子。清单 2-39 和 2-40 展示了一个使用二元语法和概率分布构建语言模型的示例。它们使用了维基百科上关于自然语言处理的几行文字。

***清单 2-39.*** 构建语言模型

```
import pandas as pd
from nltk.util import ngrams
import numpy as np
from collections import Counter
import re
```

***清单 2-40.***

`inp_text=open("nlp_text.txt","r").read()`

### 打印字符

`Inp_text[0:100]`

```
'Natural language processing\n\n\nAn automated online assistant providing
customer service on a web page'
```

你需要进行一些预处理并清理文本，以便所有单词都被规范化。你也可以根据手头的用例移除停用词。请参阅清单 2-41。

***清单 2-41.*** 清理文本

```
inp_text = inp_text.lower()
inp_text1 = re.sub('[^a-z/s]+',' ',inp_text)
```

`Inp_text1[0:1000]`

```
'natural language processing an automated online assistant providing
customer service on a web page an example of an application where natural
language processing is a major component natural language processing nlp
is a subfield of linguistics computer science information engineering and
artificial intelligence concerned with the interactions between computers
and human natural languages in particular how to program computers to
process and analyze large amounts of natural language data challenges in
```



