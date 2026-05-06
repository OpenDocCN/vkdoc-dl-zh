# Python 与自然语言工具包（NLTK）

Python 编程语言提供了多种专为解决特定自然语言处理（NLP）任务而设计的工具和库。在这个生态系统中，你会发现自然语言工具包，通常称为 NLTK，它是一个开源的库、工具和教育资源集合，旨在促进 NLP 应用程序的构建。

NLTK 包含了针对前面提到的各种 NLP 任务的库，并且还包括用于子任务的专门库，例如句子解析、分词、词干提取、词形还原（将单词还原为其词根形式的技术）以及标记化（将短语、句子、段落和篇章分解成标记，以增强计算机对文本的理解）。此外，NLTK 还包含一些库，这些库能够实现高级功能，如语义推理，即从文本来源中提取的事实中推导出逻辑结论。

接下来的段落中提供了一些特征工程的示例。

通过输入`pip install nltk`来安装最新版本的 NLTK。这对于此处所示的所有 NLTK 示例均适用。

## 标记化

```
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
text = "The quick brown fox jumped over the lazy dog."
tokens = word_tokenize(text)
print(tokens)
Output:
['The', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog', '.']
```



### 停用词移除

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
text = "The quick brown fox jumped over the lazy dog."
stop_words = set(stopwords.words('english'))
tokens = word_tokenize(text)
filtered_tokens = [token for token in tokens if not token in stop_words]
print(filtered_tokens)
Output:
['The', 'quick', 'brown', 'fox', 'jumped', 'lazy', 'dog', '.']
```

### 词干提取

```python
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
text = "The quick brown foxes jumped over the lazy dogs."
stemmer = PorterStemmer()
tokens = word_tokenize(text)
stemmed_tokens = [stemmer.stem(token) for token in tokens]
print(stemmed_tokens)
```

**输出**

```python
['the', 'quick', 'brown', 'fox', 'jump', 'over', 'the', 'lazi', 'dog', '.']
```

### 词形还原

```python
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
text = "The quick brown foxes jumped over the lazy dogs."
lemmatizer = WordNetLemmatizer()
tokens = word_tokenize(text)
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
print(lemmatized_tokens)
```

**输出：**

```python
['The', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog', '.']
```

### 用于 NLP 特征工程的 N-gram

```python
import nltk
from nltk.util import ngrams
text = "The quick brown foxes jumped over the lazy dogs."
tokens = nltk.word_tokenize(text)
bigrams = ngrams(tokens, 2) trigrams = ngrams(tokens, 3)
print(list(bigrams))
print(list(trigrams))
```

**输出：**

```python
[('The', 'quick'), ('quick', 'brown'), ('brown', 'foxes'), ('foxes', 'jumped'), ('jumped', 'over'), ('over', 'the'), ('the', 'lazy'), ('lazy', 'dogs'), ('dogs', '.')] [('The', 'quick', 'brown'), ('quick', 'brown', 'foxes'), ('brown', 'foxes', 'jumped'), ('foxes', 'jumped', 'over'), ('jumped', 'over', 'the'), ('over', 'the', 'lazy'), ('the', 'lazy', 'dogs'), ('lazy', 'dogs', '.')]
```

### 词性标注

```python
import nltk
nltk.download('averaged_perceptron_tagger')
text = "The quick brown foxes jumped over the lazy dogs."
tokens = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
print(pos_tags)
```

**输出：**

```python
[('The', 'DT'), ('quick', 'JJ'), ('brown', 'NN'), ('foxes', 'NNS'), ('jumped', 'VBD'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dogs', 'NNS'), ('.', '.')]
```

### 命名实体识别

```python
import nltk
nltk.download('words')
nltk.download('maxent_ne_chunker')
nltk.download('averaged_perceptron_tagger')
text = "John Smith works at Google in New York City."
tokens = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
ner_tags = nltk.ne_chunk(pos_tags)
print(ner_tags)
```

**输出：**

```python
(S (PERSON John/NNP) (PERSON Smith/NNP) works/VBZ at/IN (ORGANIZATION Google/NNP) in/IN (GPE New/NNP York/NNP City/NNP) ./.)
```

### TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [ "The quick brown fox jumps over the lazy dog.",
"The quick brown foxes jump over the lazy dogs and cats.",
"The lazy dogs and cats watch the quick brown foxes jump over the moon."]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
print(tfidf_matrix.toarray())
```

**输出：**

```python
[[0\. 0\. 0\. 0.51785612 0\. 0\. 0\. 0\. 0\. 0\. 0.68091856 0.51785612 0\. ]
[0\. 0\. 0\. 0.46519584 0\. 0\. 0.59817854 0\. 0\. 0\. 0\. 0.46519584 0.59817854]
[0.33682422 0.33682422 0.33682422 0.30794004 0.33682422 0.33682422 0\. 0.33682422 0.33682422 0.33682422 0\. 0.30794004 0\. ]]
```

表 1-1 列出了基于 NLTK 的一些常见自然语言处理特征提取技术。

**表 1-1** 常见的 NLP 特征提取技术

| 技术 | 主要特征 | 使用场景 | 规模与复杂度 |
| --- | --- | --- | --- |
| `CountVectorizer` | 将文本转换为词频矩阵 | 文本分类、主题建模 | 简单快速，适用于中小型数据集 |
| `TF-IDF` | 根据重要性为词语分配权重 | 信息检索、文本分类 | 更复杂且计算开销大，适用于中大型数据集 |
| 词嵌入 | 基于语义和句法的词语向量表示 | 文本分类、信息检索 | 可处理大型数据集，训练计算开销大 |
| 词袋模型 | 将文本表示为词频向量 | 文本分类、情感分析 | 简单快速，适用于中小型数据集 |
| N-gram 词袋模型 | 捕捉 n 个词语序列的频率 | 文本分类、情感分析 | 规模和复杂度取决于 n-gram 大小和数据集 |
| `Hashing Vectorizer` | 使用哈希函数将词语映射到固定大小的特征空间 | 大规模文本分类、在线学习 | 适用于大型数据集，内存高效，可能存在哈希冲突 |
| 潜在狄利克雷分配 (LDA) | 识别语料库中的主题，并为每个文档分配概率分布 | 主题建模、内容分析 | 适用于中大型数据集，计算开销大 |
| 非负矩阵分解 (NMF) | 将文档-词项矩阵分解为低维部分 | 主题建模、内容分析 | 适用于中型数据集，计算开销大 |
| 主成分分析 (PCA) | 降低文档-词项矩阵的维度 | 文本可视化、文本压缩 | 适用于大型数据集，计算开销大 |
| 词性标注 | 为文本中的每个词语分配词性标签 | 命名实体识别、文本分类 | 需要额外处理，适用于中小型数据集 |

## 词嵌入与语义理解

词嵌入是自然语言处理中的一个关键概念，它将词语转换为实值向量以进行文本分析。NLP 的这一进步显著增强了计算机理解文本的能力。这是深度学习的一大飞跃，能够有效解决复杂的 NLP 挑战。

自然语言处理面临的一个主要挑战是，它无法根据上下文解释具有多重含义的词语。上下文语义分析在澄清此类歧义、提升基于文本的 NLP 应用的精确度方面起着关键作用。让我们探讨一下 NLP 中消歧的重要性。

本节将深入探讨这两个概念。



