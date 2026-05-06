# 使用深度文本搜索的多语言搜索引擎

这些搜索引擎面临的挑战之一是多语言问题。这些产品具有很强的地域性，而英语并非唯一使用的语言。为了解决这个问题，我们可以使用深度文本搜索（Deep Text Search）。

`Deep Text Search` 是一个基于人工智能的多语言文本搜索引擎，采用了 Transformer 模型。它支持 50 多种语言。以下是它的一些特性。

-   搜索速度更快。
-   推荐结果非常准确。
-   最适合用于实现基于 Python 的应用程序。

让我们使用以下数据集来了解这个库。

-   英文数据集包含 30 行和 13 列。

[`https://data.world/login?next=%2Fpromptcloud%2Fwalmart-product-data-from-usa%2Fworkspace%2Ffile%3Ffilename%3Dwalmart_com-ecommerce_product_details__20190311_20191001_sample.csv`](https://data.world/login%253Fnext%253D%252Fpromptcloud%252Fwalmart-product-data-from-usa%252Fworkspace%252Ffile%253Ffilename%253Dwalmart_com-ecommerce_product_details__20190311_20191001_sample.csv)

-   阿拉伯语数据集是一个与阿拉伯语报纸语料库相关的文本文件。

[`https://www.kaggle.com/abedkhooli/arabic-bert-corpus`](https://www.kaggle.com/abedkhooli/arabic-bert-corpus)

-   印地语数据集包含从印地语新闻网站收集的 900 条电影评论，分为三类（正面、中性、负面）。

[`https://www.kaggle.com/disisbig/hindi-movie-reviews-dataset?select=train.csv`](https://www.kaggle.com/disisbig/hindi-movie-reviews-dataset%253Fselect%253Dtrain.csv)

-   日语数据集包含日本首相的推文。

[`https://www.kaggle.com/team-ai/shinzo-abe-japanese-prime-minister-twitter-nlp`](https://www.kaggle.com/team-ai/shinzo-abe-japanese-prime-minister-twitter-nlp)

让我们从英文数据集开始。

安装所需的包并导入库。

```
!pip install neuspell
!pip install -e neuspell/
!git clone https://github.com/neuspell/neuspell; cd neuspell
!pip install DeepTextSearch
import os
os.chdir("/content/neuspell")
!pip install -r /content/neuspell/extras-requirements.txt
!python -m spacy download en_core_web_sm
#Unzipping the multi-linguistic packages
!wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
!unzip *.zip
# importing nltk
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
#import DeepTextSearch
from DeepTextSearch import TextEmbedder,TextSearch,LoadData
from nltk.corpus import wordnet
import pandas as pd
from neuspell import BertChecker
```

让我们使用 `BERTChecker` 进行拼写检查。它也支持多种语言。

```
spellcheck = BertChecker()
spellcheck.from_pretrained(
ckpt_path=f"/content/multi_cased_L-12_H-768_A-12")
# ""
```

让我们输入查询并检查其工作方式。

```
X=input("Enter Product Name:")
y=spellcheck.correct(X)
print(y)
Enter Product Name: shirts
shirts
```

让我们也使用词性标注（POS tagging）从给定的查询中选择相关词汇。

```
#function to get the POS tag
def preprocess(sent):
sent = nltk.word_tokenize(sent)
sent = nltk.pos_tag(sent)
return sent
sent = preprocess(y)
l=[]
for i in sent:
if i[1]=='NNS' or i[1]=='NN':
l.append(i[0])
print(l)
['shirts']
```

下一步，让我们使用查询扩展来获取词汇的同义词，这样我们可以获得更相关的推荐。

```
query=""
for i in l:
query+=i
synset = wordnet.synsets(i)
query+=" "+synset[0].lemmas()[0].name()+" "
print(query)
shirts shirt
```

我们创建了这个字典，以便根据描述中的推荐来显示产品名称。

```
#importing the data
df=pd.read_csv("walmart_com-ecommerce_product_details__20190311_20191001_sample.csv")
df1=df.set_index("Description", inplace = False)
df2=df1.to_dict()
dict1=df2['Product Name']
```

将数据嵌入到一个 pickle 文件中，因为库要求数据采用这种格式。

```
data = LoadData().from_csv("walmart_com-ecommerce_product_details__20190311_20191001_sample.csv")
TextEmbedder().embed(corpus_list=data)
corpus_embedding = TextEmbedder().load_embedding()
```

基于查询搜索最相关的十个产品。

```
n=10
t=TextSearch().find_similar(query_text=query,top_n=n)
for i in range(n):
t[i]['text']=dict1[t[i]['text']]
print(t[i])
```

图 4-14 显示了针对“shirts”搜索查询的结果。

![../images/517793_1_En_4_Chapter/517793_1_En_4_Fig14_HTML.png](img/517793_1_En_4_Fig14_HTML.png)

图 4-14

模型输出

现在让我们看看搜索在阿拉伯语中是如何工作的。

```
X1=input("Search Engine:")
y1=spellcheck.correct(X1)
print(y1)
```

![../images/517793_1_En_4_Chapter/517793_1_En_4_Figc_HTML.jpg](img/517793_1_En_4_Figc_HTML.jpg)

让我们导入阿拉伯语数据语料库来执行搜索。

```
# import library and data
from DeepTextSearch import LoadData
data1 = LoadData().from_text("wiki_books_test_1.txt")
TextEmbedder().embed(corpus_list=data1)
corpus_embedding = TextEmbedder().load_embedding()
```

![../images/517793_1_En_4_Chapter/517793_1_En_4_Figd_HTML.png](img/517793_1_En_4_Figd_HTML.png)

让我们使用 `textseach` 函数查找最相关的 10 个文档。图 4-15 显示了输出结果。

```
TextSearch().find_similar(query_text=y1,top_n=10)
```

![../images/517793_1_En_4_Chapter/517793_1_En_4_Fig15_HTML.jpg](img/517793_1_En_4_Fig15_HTML.jpg)

图 4-15

模型输出

现在让我们看看搜索在印地语中是如何工作的。

```
X_hindi=input("Search Engine:")
y_hindi=spellcheck.correct(X_hindi)
print(y_hindi)
```

![../images/517793_1_En_4_Chapter/517793_1_En_4_Fige_HTML.jpg](img/517793_1_En_4_Fige_HTML.jpg)

```
#loading the Hindi data corpus
data_hindi = LoadData().from_csv("hindi.csv")
TextEmbedder().embed(corpus_list=data_hindi)
corpus_embedding = TextEmbedder().load_embedding()
```

让我们查找最相关的 10 个结果。图 4-16 显示了输出结果。

```
TextSearch().find_similar(query_text=y_hindi,top_n=10)
```

![../images/517793_1_En_4_Chapter/517793_1_En_4_Fig16_HTML.png](img/517793_1_En_4_Fig16_HTML.png)

图 4-16

模型输出

让我们再尝试一种语言，日语。

```
X_japanese=input("Search Engine:")
# y_japanese=spellcheck.correct(X_japanese)
print(X_japanese)
```

![../images/517793_1_En_4_Chapter/517793_1_En_4_Figf_HTML.png](img/517793_1_En_4_Figf_HTML.png)

```
#loading the data
data_japanese = LoadData().from_text("Japanes_Shinzo Abe Tweet 20171024 - Tweet.csv")
TextEmbedder().embed(corpus_list=data_chinese)
corpus_embedding = TextEmbedder().load_embedding()
```

根据搜索查询查找最相关的十条推文。图 4-17 显示了输出结果。

```
TextSearch().find_similar(query_text=X_japanese,top_n=10)
```

![../images/517793_1_En_4_Chapter/517793_1_En_4_Fig17_HTML.jpg](img/517793_1_En_4_Fig17_HTML.jpg)

图 4-17

模型输出



## 总结

在本章中，我们使用多种模型实现了一个搜索引擎和推荐系统。我们从一个简单的推荐系统开始，采用`TF-IDF`方法计算所有产品描述的相似度得分。根据描述，产品被排序并展示给用户。随后，我们探索了如何使用词嵌入构建一个简单的搜索引擎，并对结果进行排序。

接着，我们深入研究了诸如`PyTerrier`和`Sentence-BERT`等高级模型，这些模型利用预训练模型提取向量。由于这些模型基于深度学习，其结果相比传统方法要好得多。我们还使用了`Deep Text Search`，这是另一个适用于多语言文本语料库的深度学习库。

