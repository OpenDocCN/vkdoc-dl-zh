# 5. 创建简历解析、筛选与入围系统

本项目的目标是利用自然语言处理创建一个简历入围系统。

## 背景

面对数百万求职者，要从成百上千份简历中筛选出最佳应聘者几乎是不可能的。大多数简历没有标准格式；也就是说，市面上几乎每份简历都有其独特的结构和内容。人力资源人员必须亲自审阅每份简历，以确定最符合职位描述的人选。这一过程既耗时又容易出错，因为合适的候选人可能会在此过程中被忽略。

企业目前面临的挑战之一，是在有限的时间和资源下为职位挑选合适的候选人。在挑选最佳候选人时，招聘人员必须考虑以下几点：

-   从数据库中的简历集合里，手动逐一筛选，找出与给定职位描述最匹配的简历。
-   在选出的简历中，根据与职位描述的相关性进行排序。
-   排序完成后，需要提取姓名、联系电话和电子邮件地址等信息，以便进一步推进候选人流程。

为了克服这些挑战，让我们看看 NLP 如何构建一个简历解析和入围系统，该系统能够根据职位描述（JD）从海量简历中解析、匹配、筛选和排序。

## 方法与步骤

解决此项目的关键需求是职位描述和简历，它们作为模型的输入，然后根据格式（Word 或 PDF）传递给文档读取器，该读取器从 Word/PDF 文档中提取所有文本内容。

现在我们有了文本数据，对其进行处理以去除噪声和不需要的信息（如停用词和标点符号）就变得至关重要。

一旦简历和职位描述处理完毕，我们需要使用计数向量化器或`TD-IDF`将文本转换为特征。对于这个用例，`TD-IDF`可能更有意义，因为我们希望强调简历中提到的每个关键词（技能）的出现频率。之后，我们使用*截断奇异值分解*来降低特征向量的维度。

然后我们进入模型构建阶段，在此阶段我们有一个相似度引擎，用于捕获两个文档（简历和职位描述）在关键词方面的相似程度。利用相似度得分，我们对系统中每个职位描述对应的每份简历进行排序。例如，有两个职位描述（数据科学和分析）以及 50 份来自不同技能领域的简历。那么，在这 50 份简历中，该模型将为数据科学和分析分别提供排名靠前的简历。

模型构建完成，并且我们为给定的职位描述选出了排名靠前的简历后，提取相关信息（如姓名、联系电话、地点和电子邮件地址）就变得至关重要。

最后，我们汇总所有结果，并以表格形式呈现，招聘人员可以访问该表格并继续进行招聘流程。验证和可视化层有助于招聘人员进一步验证，并在无需打开简历了解候选人背景的情况下，查看简历中的核心技能。

图 5-1 展示了解决此问题的方法流程图。

![../images/517793_1_En_5_Chapter/517793_1_En_5_Fig1_HTML.jpg](img/517793_1_En_5_Fig1_HTML.jpg)

图 5-1

展示了模型的完整框架

**注意** 信息与实体的提取是在原始文本（未经预处理）上进行的。

## 实现

我们考虑的数据集包含来自不同领域的 32 份不同简历和三个不同的职位描述。该数据集是开源的。这里使用的数据集包含不同文件格式的简历和职位描述。以下是数据集的链接。（本章演示仅考虑了部分数据集，但每个领域档案都有数千份简历。）

数据收集完成后，让我们开始实现。

### 安装与导入所需库

```
# 安装所需库
!pip install textract
!pip install -U nltk
!pip install pdfminer3
!pip install mammoth
!pip install locationtagger
# 导入所需库
import pandas as pd
from google.colab import drive
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import en_core_web_sm
nlp = en_core_web_sm.load()
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mammoth
import locationtagger
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.corpus import wordnet
nltk.download('wordnet')
from sklearn.decomposition import TruncatedSVD
```



### 阅读简历与职位描述

现在，让我们从创建简历和职位描述目录的路径开始。

```
# 创建简历和职位描述目录
directory = '/content/drive/MyDrive/'
resume_path = directory + 'Resumes/'
jd_path = directory + 'JD/'
```

接下来，让我们从简历和职位描述中提取所有文本信息。我们来编写一个函数，用于提取 PDF 格式的简历。它也可以提取表格中的数据。这里我们使用了 `pdfminer3` 中定义的 `PDFResourceManager`、`PDFPageInterpreter` 和 `TextConverter` 函数。

```
#此函数用于从 PDF 文件中提取文本。它也可以从 PDF 文件中提取表格
def pdf_extractor(path):
r_manager = PDFResourceManager()
output = io.StringIO()
converter = TextConverter(r_manager, output, laparams=LAParams())
p_interpreter = PDFPageInterpreter(r_manager, converter)
with open(path, 'rb') as file:
for page in PDFPage.get_pages(file,caching=True,check_extractable=True):
p_interpreter.process_page(page)
text = output.getvalue()
converter.close()
output.close()
return text
```

以下函数用于读取以下任意格式的文档。

*   PDF
*   DOCX
*   DOC
*   TXT

```
# 用于读取 pdf、docx、doc 和 txt 文件的函数
def read_files(file_path):
fileTXT = []
# 此 for 循环用于读取函数中 file_path 指定的所有文件
for filename in os.listdir(file_path):
# 如果文档是 pdf 格式，则执行此代码
if(filename.endswith(".pdf")):
try:
fileTXT.append(pdf_extractor(file_path+filename)) # 此处使用 pdf_extractor 函数提取 pdf 文件
except Exception:
print('读取 pdf 文件时出错 :' + filename)
# 如果文档是 docx 格式，则执行此代码
if(filename.endswith(".docx")):
try:
with open(file_path + filename, "rb") as docx_file:
result = mammoth.extract_raw_text(docx_file)
text = result.value
fileTXT.append(text)
except IOError:
print('读取 .docx 文件时出错 :')
# 如果给定文档是 doc 格式，则执行此循环
if(filename.endswith(".doc")):
try:
text = textract.process(file_path+filename).decode('utf-8')
fileTXT.append(text)
except Exception:
print('读取 .doc 文件时出错 :' + filename)
# 如果给定文件是 txt 格式，则执行此文件
if(filename.endswith(".txt")):
try:
myfile = open(file_path+filename, "rt")
contents = myfile.read()
fileTXT.append(contents)
except Exception:
print('读取 .txt 文件时出错 :' + filename)
return fileTXT
```

`resumeTxt` 是一个包含所有候选人简历的列表。

```
# 调用 read_files 函数读取所有简历
resumeTxt = read_files(resume_path)
# 显示第一份简历
resumeTxt[0]
```

图 5-2 显示了输出结果。

![../images/517793_1_En_5_Chapter/517793_1_En_5_Fig2_HTML.jpg](img/517793_1_En_5_Fig2_HTML.jpg)

图 5-2

显示 `resumeTxt` 的输出结果

`jdTxt` 是所有职位描述的列表。

```
# 调用 read_files 函数读取所有职位描述
jdTxt = read_files(jd_path)
# 显示第一份职位描述
jdTxt[0]
```

图 5-3 显示了输出结果。

![../images/517793_1_En_5_Chapter/517793_1_En_5_Fig3_HTML.jpg](img/517793_1_En_5_Fig3_HTML.jpg)

图 5-3

显示 `jdTxt` 的输出结果

### 文本处理

你必须处理文本以去除噪声和其他无关信息。让我们遵循一些基本的文本清洗步骤，例如将所有单词转换为小写、移除特殊字符（`%`、`$`、`#` 等）、剔除包含数字的单词（例如 `hey199` 等）、删除多余空格、移除停用词（例如重复出现的单词，如 *is*、*the*、*an* 等）等等。

```
# 此函数帮助我们移除停用词、标点符号、特殊字符、多余空格和数字数据。它还将所有文本转换为小写。
Def preprocessing(Txt):
sw = stopwords.words('english')
space_pattern = '\s+'
special_letters = "[^a-zA-Z#]"
p_txt = []
for resume in Txt:
text = re.sub(space_pattern, ' ', resume) # 移除多余空格
text = re.sub(special_letters, ' ', text) # 移除特殊字符
text = re.sub(r'[^\w\s]','',text) # 移除标点符号
text = text.split() # 将文本中的单词拆分
text = [word for word in text if word.isalpha()] # 仅保留字母单词
text = [w for w in text if w not in sw] # 移除停用词
text = [item.lower() for item in text] # 将单词转换为小写
p_txt.append(" ".join(text)) # 将所有单词重新连接
return p_txt
```

`p_résuméTxt` 包含所有预处理后的简历。

```
# 调用 preprocessing 函数来清洗所有简历
p_resumeTxt = preprocessing(resumeTxt)
# 显示第一份预处理后的简历
p_resumeTxt[0]
```

图 5-4 显示了输出结果。

![../images/517793_1_En_5_Chapter/517793_1_En_5_Fig4_HTML.jpg](img/517793_1_En_5_Fig4_HTML.jpg)

图 5-4

显示 `p_résuméTxt` 的输出结果

`jds` 包含所有预处理后的职位描述。

```
# 调用 preprocessing 函数来清洗所有职位描述
jds = preprocessing(jdTxt)
# 显示第一份预处理后的职位描述
Jds[0]
```

图 5-5 显示了输出结果。

![../images/517793_1_En_5_Chapter/517793_1_En_5_Fig5_HTML.jpg](img/517793_1_En_5_Fig5_HTML.jpg)

图 5-5

显示 `JDs` 的输出结果

你可以清楚地看到预处理文本与原始文本之间的差异。一旦有了处理后的文本，将其转换为机器可以理解的特征就很重要了。

### 文本到特征

TF-IDF 衡量一个词对特定文档的重要程度。

这里我们使用 `sklearn` 中的默认 `TfidfVectorizer` 库。

```
# 合并简历和职位描述以计算 TF-IDF 和余弦相似度
TXT = p_resumeTxt+jds
# 计算所有简历和职位描述的 TF-IDF 分数
tv = TfidfVectorizer(max_df=0.85,min_df=10,ngram_range=(1,3))
# 将 TF-IDF 转换为 DataFrame
tfidf_wm = tv.fit_transform(TXT)
tfidf_tokens = tv.get_feature_names()
df_tfidfvect1 = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
print("\nTD-IDF 向量化器\n")
print(df_tfidfvect1[0:10])
```

图 5-6 显示了输出结果。

![../images/517793_1_En_5_Chapter/517793_1_En_5_Fig6_HTML.jpg](img/517793_1_En_5_Fig6_HTML.jpg)

图 5-6

TF-IDF 的输出结果

根据文档的大小，这可能会产生一个巨大的矩阵，因此使用特征降维技术来降低维度非常重要。

### 特征降维

截断*奇异值分解*（SVD）是最著名的降维方法之一。它将矩阵 `M` 分解为三个矩阵 `U`、`Σ` 和 `V^T` 以生成特征。`U`、`Σ` 和 `V^T` 描述如下。

*   `U` 是左奇异矩阵
*   `Σ` 是对角矩阵
*   `V^T` 是右奇异矩阵

截断 SVD 产生一种分解，我们可以指定想要的列数，从而保留最相关的特征。

```
# 定义变换
dimrec = TruncatedSVD(n_components=30, n_iter=7, random_state=42)
transformed = dimrec.fit_transform(df_tfidfvect1)
# 将变换后的向量转换为列表
vl = transformed.tolist()
# 将列表转换为 DataFrame
fr = pd.DataFrame(vl)
print('SVD 特征向量')
print(fr[0:10])
```

图 5-7 显示了输出结果。

![../images/517793_1_En_5_Chapter/517793_1_En_5_Fig7_HTML.jpg](img/517793_1_En_5_Fig7_HTML.jpg)

图 5-7

SVD 的输出结果



## 模型构建

这是一个相似度度量问题，系统会向招聘人员推荐前 N 个（N 可自定义）最匹配的简历。为此，我们采用余弦相似度，它用于衡量两个向量的相似程度。

公式如下：

![$$ \mathit{\cos}\left(x,y\right)=\frac{\ x\bullet y}{\left|\left|x\right|\right|\ast \left|\left|y\right|\right|} $$](img/517793_1_En_5_Chapter_TeX_Equa.png)

- `x` ⋅ `y` 是向量 `x` 和 `y` 的点积
- `||x||` 和 `||y||` 是向量 `x` 和 `y` 的长度
- `||x||` ∗ `||y||` 是向量 `x` 和 `y` 的叉积

```
# 计算职位描述与简历之间的余弦相似度，以找出最适合某职位描述的简历
similarity = cosine_similarity(df_tfidfvect1[0:len(resumeTxt)],df_tfidfvect1[len(resumeTxt):])
# 职位描述的列名
abc = []
for i in range(1,len(jds)+1):
abc.append(f"JD {i}")
# 相似度得分的 DataFrame
Data=pd.DataFrame(similarity,columns=abc)
print('\n 余弦相似度\n')
print(Data[0:10])
```

图 5-8 显示了输出结果。

![../images/517793_1_En_5_Chapter/517793_1_En_5_Fig8_HTML.jpg](img/517793_1_En_5_Fig8_HTML.jpg)

图 5-8

输出结果展示了每份简历（代表每一行）与相应职位描述之间的相似度得分。

