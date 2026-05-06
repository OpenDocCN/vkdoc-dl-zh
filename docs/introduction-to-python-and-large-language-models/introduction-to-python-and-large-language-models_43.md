# 文本摘要与文档理解

在大数据时代，海量文本数据凸显了高效文本摘要方法的关键意义。文本摘要技术能将冗长文档或文章提炼为清晰简洁的摘要，同时保留核心含义与关键信息，在信息检索、内容生成等多个领域具有重要价值。摘要已成为自然语言处理应用中不可或缺的组成部分。

文本摘要是自然语言处理中的核心任务，旨在将大量文本压缩为更短但连贯的表述，同时保留关键信息。

文本摘要主要有三种方法：*直接式*、*抽象式*和*抽取式*摘要。

- **直接式摘要：** 这是最直接的方法，直接要求大语言模型提供文档摘要。虽然快速简便，但处理长文本或复杂文本时可能遇到困难。
- **抽象式摘要：** 通过生成包含源文本中未直接出现的词语、短语或句子的简洁摘要。该方法依赖上下文理解与类人语言生成能力来阐述核心思想。借助大语言模型等先进技术，抽象式摘要技术旨在将内容提炼并改写为更简洁的形式。
- **抽取式摘要：** 直接从源文本中选择并提取最相关的句子或短语组成摘要。该方法不涉及重写或生成新句子。抽取式摘要采用句子评分、排序等技术来定位并提取最重要的内容。



### 使用用户提供的 URL 进行文章摘要的应用

```
from langchain.output_parsers import PydanticOutputParser
from pydantic import field_validator
from pydantic import BaseModel, Field
from typing import List
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import requests
from newspaper import Article
openai_key = 'YOUR-API-KEY'
headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}
article_url = input("Enter Article URL To Be Summarized: ")
session = requests.Session()
try:
response = session.get(article_url, headers=headers, timeout=10)
if response.status_code == 200:
article = Article(article_url)
article.download()
article.parse()
else:
print(f"Failed to fetch article at {article_url}")
except Exception as e:
print(f"Error occurred while fetching article at {article_url}: {e}")
article_title = article.title
article_text = article.text
# create output parser class
class ArticleSummary(BaseModel):
title: str = Field(description="Title of the article")
summary: List[str] = Field(description="Bulleted list summary of the article")
# validating whether the generated summary has at least three lines
@field_validator('summary')
def has_three_or_more_lines(cls, list_of_lines):
if len(list_of_lines) < 3:
raise ValueError("Generated summary has less than three bullet points!")
return list_of_lines
# set up output parser
parser = PydanticOutputParser(pydantic_object=ArticleSummary)
# create prompt template
# notice that we are specifying the "partial_variables" parameter
template = """
You are an experienced content writing assistant that summarizes online articles.
Here's the article you want to summarize.
==================
Title: {article_title}
{article_text}
==================
{format_instructions}
"""
prompt = PromptTemplate(
template=template,
input_variables=["article_title", "article_text"],
partial_variables={"format_instructions": parser.get_format_instructions()}
)
# Format the prompt using the article title and text obtained from scraping
formatted_prompt = prompt.format_prompt(article_title=article_title, article_text=article_text)
# Instantiate model class
model = OpenAI(openai_api_key = openai_key, model_name="gpt-3.5-turbo-instruct", temperature=0.0)
# Use the model to generate a summary
output = model(formatted_prompt.to_string())
# Parse the output into the Pydantic model
parsed_output = parser.parse(output.split("\"]}")[0] + "\"]}")
print("Title:", parsed_output.title)
print("Article Summary:")
for item in parsed_output.summary:
print("-", item)
```

**示例输出：**

```
Enter Article URL To Be Summarized: https://www.techtarget.com/whatis/definition/large-language-model-LLM
Title: What are Large Language Models?
Article Summary:
- LLMs are becoming increasingly important to businesses as AI continues to grow and dominate the business setting.
- LLMs take a complex approach involving multiple components, including training on large volumes of data and using self-supervised learning and deep learning techniques.
- LLMs have a wide range of uses, including text generation, translation, content summary, rewriting, classification, sentiment analysis, and conversational AI.
- LLMs offer numerous advantages, such as extensibility, flexibility, performance, accuracy, and ease of training.
- However, there are also challenges and limitations to using LLMs, such as high development and operational costs, potential bias, lack of explainability, hallucination, complexity, and vulnerability to glitch tokens.
```

**说明**

此应用是上一章中使用 OpenAI GPT-3.5 模型进行在线文章摘要应用的扩展版本。在使用之前，你需要安装以下软件包：

*   `Langchain==0.1.4`
*   `deeplake==3.9.11`
*   `Openai==1.10.0`
*   `Tiktoken==0.7.0`
*   `Newspaper3k==0.2.8`
*   `Pydantic==2.7.4`

你可以通过在笔记本或终端中执行以下命令来完成安装：`pip install langchain==0.1.4 deeplake openai==1.10.0 tiktoken newspaper3k pydantic`。你还需要一个 OpenAI 的 API 密钥。

**以下是代码运行过程的详细说明：**

1.  它导入了必要的模块和类，包括 `PydanticOutputParser`、`BaseModel`、`Field`、`List`、`PromptTemplate`、`OpenAI`、`requests` 和 `Article`。
2.  它提示用户输入待摘要文章的 URL。
3.  它发起请求，从提供的 URL 获取文章内容，并使用 `newspaper` 库进行解析。
4.  它定义了一个名为 `ArticleSummary` 的 Pydantic 模型类，包含两个字段：`title` 和 `summary`。`summary` 字段通过一个验证器 `has_three_or_more_lines` 来确保生成的摘要至少包含三个要点。
5.  它为 `ArticleSummary` 类设置了一个使用 `PydanticOutputParser` 的输出解析器。
6.  它使用从抓取中获得的文章标题和文本来定义提示模板。
7.  它使用文章标题和文本来格式化提示。
8.  它使用指定的 API 密钥和模型名称实例化 OpenAI 模型类。
9.  它使用 OpenAI 模型根据格式化后的提示生成摘要。
10. 它将输出解析为 `ArticleSummary` Pydantic 模型。
11. 它打印解析后的输出，显示文章的标题和摘要。

**注意**

OpenAI 摘要

*实际的摘要生成发生在 OpenAI 模型内部，输入提示与文章详情一起提供，模型据此生成摘要。Pydantic 模型用于确保输出摘要的结构和验证。*

## 问答系统：知识触手可及

### 通过大型语言模型（LLM）增强问答能力

大型语言模型（LLM）的出现，例如 OpenAI 的 GPT，彻底革新了问答（QA）领域。这些模型通过促进对查询及其上下文细微差别的深刻理解，将生成式问答提升到了新的高度。LLM 能够构建类似人类对话的交互，生成与用户潜在意图高度一致的响应。此外，它们还擅长根据检索到的上下文提供智能且富有洞察力的摘要。

在最基本的层面上，一个简单的生成式问答系统只需要用户的文本查询和一个 LLM。然而，更复杂的系统可能会在特定的领域知识上进行额外的训练。它们还可以与搜索和推荐引擎集成，不仅回答问题，还能无缝链接相关信息来源。通过这些进步，这些模型超越了传统的搜索范式，向着更动态、响应更快、交互性更强的信息检索模式发展。

### 利用大型语言模型进行高级文档分析

将生成式问答模型集成到信息检索系统中，标志着文档分析领域的重大进步。利用大型语言模型（LLM）可以创建更动态、更直观的系统，这些系统不仅能搜索数据，还能理解和解释数据。这种集成彻底改变了跨不同领域处理复杂查询的方式，从根本上改变了信息交互和检索的方式。

### 从数据到响应的旅程：全面概述



#### 文档解析与准备

该过程始于加载并解析各种格式的文档，例如文本、PDF 或数据库条目。借助 `NLTK` 等自然语言处理（NLP）包，文档被分割成从段落到句子的可管理单元。这些工具简化了任务处理，并妥善处理了换行符和特殊字符等细节，使工程师能够专注于更高级的方面。

#### 文本嵌入与索引

每个文本单元都会从字符格式转换为数值向量，生成包含语义信息的嵌入。诸如 `Universal Sentence Encoder`、`DRAGON+` 和 `Instructor` 等模型被用于生成定制化的嵌入，这些嵌入可以基于特定提示，也可以利用大型语言模型的能力。

这些嵌入被存储在向量数据库中，形成一个可搜索的索引，以便高效检索信息。`NumPy`、`Faiss` 或 `Elasticsearch`/`OpenSearch` 等工具有助于此过程，支持多种算法来构建索引和管理检索任务。

#### 查询处理与上下文检索

当用户提交查询后，系统会应用嵌入技术来匹配已索引的数据，并基于余弦相似度等相似性度量进行上下文检索。

系统会检索出最相关的文本单元，为生成答案建立上下文框架。

#### 答案生成

作为生成式模型，大型语言模型（LLM）利用检索到的上下文以及查询来构建回答。通过计算词序列的条件概率，它能生成上下文准确且富有洞察力的答案。

这种系统性的方法突显了生成式问答（GQA）系统——在增强型 LLM 的支持下——不仅能够检索相关信息，还能生成加深对查询理解的回答。这些系统代表了信息检索领域的下一次演进，促进了类似与知识渊博的实体对话的交互式交流，能够针对特定问题提供精确且富有洞察力的答案。

### 生成式问答的实际应用与用例

生成式问答（GQA）利用先进的人工智能，为用户查询提供详细且具有上下文感知能力的回答，从而增强了各种实际应用。GQA 有助于内容创作，能够自动生成信息丰富的文章和报告。这种多功能性展示了 GQA 在多个领域的变革潜力，推动了效率提升和创新。

#### 通过自动回复增强客户支持

生成式问答通过实现自动化且具有上下文感知能力的回复，正在重塑客户支持领域。借助 LLM，客户支持系统能够提供针对单个查询细微差别的准确答案，通过更快的响应时间和减少对人工客服处理常规查询的依赖，简化了支持流程。

#### 在报告和非结构化文档中的高效搜索

GQA 正在革新组织内部的文档搜索流程，尤其是在处理制造报告、物流记录和销售记录等复杂文档方面。通过集成向量数据库，可以实现对相关信息的高效索引和检索，为员工提供类似人类的交互体验。这种方法确保了从生产数据分析到客户关怀洞察等各方面信息的快速精准获取。

#### 大型组织的知识管理

对于拥有庞大知识库的大型组织而言，生成式问答系统提供了显著优势。通过构建内部知识源的全面索引，简化了信息检索过程，并利用从最新数据中得出的洞察来辅助决策。无论是查询公司政策、历史记录还是项目报告，GQA 系统都提供了一种高效的知识管理和检索方式，从而简化组织运营。

### 带来源的文档问答聊天机器人

让我们深入探讨高级人工智能应用领域，构建一个复杂的问答（QA）聊天机器人，它能够处理文档并为回答提供来源。我们的问答聊天机器人利用一种称为 `RetrievalQAWithSourcesChain` 的特殊机制，使其能够细致地浏览文档库，识别相关信息来回答查询。

该链会组织结构化的提示，引导底层语言模型生成回答。这些提示经过精心设计，以引导语言模型的输出，从而提高所提供答案的精确度和相关性。此外，检索链经过精密设计，能够细致地追溯其检索信息的来源，从而具备用可信引用支持回答的能力。

**在我们的旅程中，我们将掌握以下技巧：**

1.  抓取用户提供的在线文章，并将文本内容及其对应的 URL 存档。

2.  使用嵌入模型计算这些文档的嵌入，并将其存档在向量数据库 Deep Lake 中。

3.  将文章文本分割成易于管理的片段，同时仔细记录每个片段的来源。

4.  使用 `RetrievalQAWithSourcesChain` 来构建一个聊天机器人，它既能检索答案，又能同时追踪答案的来源。

5.  利用该链制定对查询的回答，同时呈现答案及其佐证来源。

在继续之前，请确保已通过执行以下命令安装了必要的软件包：

```
pip install langchain==0.1.4 deeplake==3.9.11 openai==0.27.8 tiktoken==0.7.0 newspaper3k==0.2.8
```

此命令将安装所需的软件包，包括 `langchain` 版本 `0.1.4`、`deeplake`、`openai` 版本 `0.27.8` 和 `tiktoken`。此外，它还会安装 `newspaper3k` 软件包版本 `0.2.8`。

接下来，务必将你的 OpenAI 和 Deep Lake API 密钥添加到环境变量中。然后，`LangChain` 库将访问这些令牌并将其用于集成：

```
import os
os.environ["OPENAI_API_KEY"] = ""
os.environ["ACTIVELOOP_TOKEN"] = ""
```



#### 抓取用户提供的文章

首先，我们来收集一些关于人工智能新闻的文章。我们的重点是获取每篇文章的文本内容及其对应的发布网址。

在代码中，你将看到以下组成部分：

1.  **导入模块：** 我们首先导入必要的 Python 库。`requests` 用于发送 HTTP 请求，`newspaper` 在从网页中提取和组织文章方面非常有用，而 `time` 则帮助我们在网络抓取过程中加入暂停。

2.  **请求头：** 某些网站可能会屏蔽缺少正确 `User-Agent` 请求头的请求，将其视为机器人活动。因此，我们定义一个 `User-Agent` 字符串来模拟真实的浏览器请求。

3.  **文章网址：** 一系列在线文章的网址列表。

4.  **网络抓取：** 通过 `requests.Session()` 建立一个 HTTP 会话，使我们能够在同一会话中执行多个请求。此外，我们初始化一个空列表 `pages_content`，用于存储抓取到的文章。

```python
import requests
from newspaper import Article # https://github.com/codelucas/newspaper
import time
headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}
article_urls = [
"https://www.site.com/2023/05/16/page-one/",
"https://www.site.com/2023/05/16/page-two/",
"https://www.site.com/2023/05/16/page-three/",
添加你的网址…
]
session = requests.Session()
pages_content = [] # 用于保存抓取的文章
for url in article_urls:
try:
time.sleep(2) # 暂停两秒以实现温和抓取
response = session.get(url, headers=headers, timeout=10)
if response.status_code == 200:
article = Article(url)
article.download()
article.parse()
pages_content.append({ "url": url, "text": article.text })
else:
print(f"获取文章失败：{url}")
except Exception as e:
print(f"获取文章时出错：{url}: {e}")
```

接下来，我们将使用一个嵌入模型来计算文档的嵌入向量，并将其保存到 Deep Lake（一个能够处理多模态向量的数据库）中。`OpenAIEmbeddings` 将作为生成文档向量表示的工具。

这些嵌入向量由高维向量组成，能够很好地封装文档的语义本质。在初始化 Deep Lake 类的实例时，我们提供一个以 `hub://...` 开头的路径，用于指定数据库名称，该数据库随后将托管在云端。

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
my_activeloop_org_id = "你的组织 ID - 通常是你的用户名"
my_activeloop_dataset_name = "qabot_with_source"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)
```

这一部分对于配置系统以根据语义内容存储和检索文档至关重要。这种功能是后续阶段的关键，其目标是找出最相关的文档来回答用户的问题。

随后，我们将把这些文章分割成更小的片段，每个片段对应的网址将作为参考点保存下来。这种分割有助于简化数据处理，使检索任务更易于管理，并在回答问题时将注意力集中在最相关的文本片段上。

`RecursiveCharacterTextSplitter` 被实例化，其块大小为 1000 个字符，相邻块之间有 100 个字符的重叠。`chunk_size` 参数定义了每个文本片段的长度，而 `chunk_overlap` 则指定了相邻片段之间共享的字符数。对于 `pages_content` 中的每个文档，文本都使用 `.split_text()` 方法进行分割。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_texts, all_metadatas = [], []
for d in pages_content:
chunks = text_splitter.split_text(d["text"])
for chunk in chunks:
all_texts.append(chunk)
all_metadatas.append({ "source": d["url"] })
```

在元数据字典中，我们使用 `source` 键来符合 `RetrievalQAWithSourcesChain` 类的预期，该类会自动从元数据中检索这个 `source` 项。随后，我们将这些分割后的块及其对应的元数据添加到 Deep Lake 数据库中。

```python
db.add_texts(all_texts, all_metadatas)
```

让我们进入构建问答聊天机器人的激动人心的阶段。我们将着手开发一个 `RetrievalQAWithSourcesChain`，这是一个不仅能够检索相关文档片段来回答问题，还能记录这些文档来源的链。

##### 启动链的设置

首先，我们使用 `from_chain_type` 方法实例化一个 `RetrievalQAWithSourcesChain`。该方法需要以下参数：

1.  `LLM`：此参数需要一个模型实例，例如 GPT-3，其温度为 0。温度参数控制模型输出的随机程度——温度越高，随机性越大；温度越低，输出越确定。

2.  `chain_type="stuff"`：此参数定义了所使用的链的类型，影响模型如何处理检索到的文档并生成响应。

3.  `retriever=db.as_retriever()`：此步骤建立了负责从 Deep Lake 数据库中获取相关文档的检索器。在这里，Deep Lake 数据库实例（称为 `db`）通过其 `as_retriever` 方法被转换为检索器。

```python
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
chain_type="stuff",
retriever=db.as_retriever())
```

最后，我们将使用这个链来生成对问题的回答。这个响应将包含问题的答案及其相关来源。

```python
d_response = chain({"question": "杰弗里·辛顿对近期人工智能趋势有何看法？"})
print("响应：")
print(d_response["answer"])
print("来源：")
for source in d_response["sources"].split(", "):
print("- " + source)
```



### 应用的完整代码

```python
import requests
from newspaper import Article # https://github.com/codelucas/newspaper
import time
headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}
article_urls = [
"https://www.site.com/2023/05/16/page-one/",
"https://www.site.com/2023/05/16/page-two/",
"https://www.site.com/2023/05/16/page-three/",
Add Your URLs…
]
session = requests.Session()
pages_content = []
for url in article_urls:
try:
time.sleep(2) # 休眠两秒，实现温和抓取
response = session.get(url, headers=headers, timeout=10)
if response.status_code == 200:
article = Article(url)
article.download()
article.parse()
pages_content.append({ "url": url, "text": article.text })
else:
print(f"获取文章失败：{url}")
except Exception as e:
print(f"获取文章时出错：{url}，错误信息：{e}")
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
my_activeloop_org_id = "Your Username"
my_activeloop_dataset_name = "langchain_course_qabot_with_source"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_texts, all_metadatas = [], []
for d in pages_content:
chunks = text_splitter.split_text(d["text"])
for chunk in chunks:
all_texts.append(chunk)
all_metadatas.append({ "source": d["url"] })
db.add_texts(all_texts, all_metadatas)
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
chain_type="stuff",
retriever=db.as_retriever())
d_response = chain({"question": "Your Question?"})
print("响应：")
print(d_response["answer"])
print("来源：")
for source in d_response["sources"].split(", "):
print("- " + source)
```

**示例输出：**

```text
我是否应该为分页页面添加 nofollow？
响应：
建议对分页序列中的所有页面使用相同的标题和描述，并为每个页面提供唯一的 URL。同时建议避免将分页序列的第一页作为规范页面，并避免索引带有过滤器或替代排序顺序的 URL。无需为分页页面添加 nofollow。
来源：
https://developers.google.com/search/docs/specialty/ecommerce/pagination-and-incremental-page-loading
```

## 聊天机器人与虚拟助手

GPT-4 等先进语言模型的出现彻底革新了聊天机器人的设计，将其边界推向了前所未有的高度。理解聊天机器人的不同类别、驱动其设计的方法论，并遵循最佳实践，对于打造真正高效的 AI 驱动助手至关重要。

教育领域正在经历持续的变革，技术在其中扮演着推动进步的关键角色。教育技术的一个前沿趋势是将聊天机器人整合到法学硕士（LLM）课程中。这些数字助手能够熟练地即时响应用户的询问，并促进信息的快速获取，展现了其增强学习体验的潜力。

### 聊天机器人的核心理念是什么？

聊天机器人是旨在模拟人类互动，同时透明地作为自动化服务运行的 AI 驱动解决方案。它们促进企业与客户之间的实时互动，处理询问、完成任务和处理交易。通过这种方式，它们使客户服务团队能够专注于解决更复杂的问题。

聊天机器人的历史可追溯至 20 世纪 60 年代，拥有丰富的发展历程，著名的前身包括 ELIZA 和 Siri。其功能依赖于多种方法，从预定义模板和关键词识别，到自然语言处理（NLP）和机器学习（ML）等复杂技术，使其能够有效理解和响应用户输入。

聊天机器人擅长处理广泛的任务，从提供客户支持到大规模高效执行重复性任务。此外，它们还能在需要时无缝地将对话转接给人工客服。聊天机器人提供灵活的全渠道体验，可在网站、即时通讯应用、虚拟助手甚至传统电话系统等多种平台上运行。

### 经 LLM 训练的聊天机器人的实际应用

利用 LLM 可以训练针对特定功能的聊天机器人，从而在各个部门开辟一系列部署机会。

*   **销售与市场营销：** 利用经 LLM 训练的聊天机器人，可以通过引导潜在客户、提供定制化推荐以及构建详细的客户画像来简化销售流程，从而增强销售和营销工作。

*   **内容营销：** 这些具备 LLM 知识的聊天机器人能够策划定制化内容、提供个性化推荐、自动化内容分发并征求客户反馈，从而强化内容营销策略。

*   **客户支持：** 借助 LLM 的能力，聊天机器人擅长自主管理客户咨询，确保持续提供准确回复，处理常见问题，并提供实时信息以丰富客户支持互动。

*   **社交媒体营销与潜在客户生成：** 融入 LLM 洞察的聊天机器人可以高效收集用户数据，参与社交媒体对话，并让客户及时了解最新发布、活动和促销信息，从而增强社交媒体营销活动和潜在客户生成工作。

*   **培训与发展：** 将经 LLM 训练的聊天机器人整合到培训计划中，可确保学习者无间断地获取学习材料、获得个性化学习路径、全天候解决疑问并即时获得反馈，从而彻底改变培训与发展策略。

### 使用 LLM 构建聊天机器人指南

使用大型语言模型（LLM）构建聊天机器人，为创建复杂、响应迅速且智能的对话代理提供了一条途径。关键方面包括模型选择与微调、数据预处理、集成与部署（部署不在本书讨论范围内），以及确保隐私与安全。通过遵循这些指南，开发者可以创建出能提供高质量用户体验、有效自动化客户服务并增强跨各种应用场景用户参与度的聊天机器人。

#### 模型选择

初始步骤至关重要——选择合适的语言模型。从 GPT-3、GPT-Neo、GPT-2 到 BERT，选择众多，每个模型在规模、能力和预训练权重上都有所不同。选择与聊天机器人的预期用途和可用资源相协调的模型。

#### 数据预处理与清洗

精心准备用于训练聊天机器人的数据集。剔除对话数据中的噪声和无关信息。数据预处理确保数据集经过精炼，有助于模型有效学习并生成连贯的回复。

#### 模型微调

接下来是在纯净的对话数据集上对所选语言模型进行微调。这涉及调整模型参数以适应特定的对话上下文，从而增强其回复生成能力。



#### 集成与部署

将微调后的模型无缝集成到聊天机器人框架或平台中。利用 `Hugging Face Transformers`、`TensorFlow` 或 `PyTorch` 等开源框架实现高效的模型集成。将聊天机器人部署到网站、即时通讯应用或 API 等多种平台，以实现广泛的可访问性。

#### 最佳实践与注意事项

- **性能评估：** 定期评估聊天机器人的性能，确保其回答与用户查询相关且符合预期语境。
- **隐私与安全措施：** 实施稳健的安全协议以保护用户数据，尤其是在处理敏感信息时，从而建立信任和保密性。
- **持续学习与改进：** 使聊天机器人能够从用户交互中学习，根据不断变化的对话模式动态调整和优化其回答，从而确保持续改善用户体验。

### 客户支持问答聊天机器人

首先，我们从在线文章中收集内容，将其分割成较小的段落，然后计算它们的嵌入向量并存储在 `Deep Lake` 中。接着，我们利用用户查询从 `Deep Lake` 中提取最相关的段落，将它们组合成提示词，供大语言模型生成最终回答。

必须认识到，使用大语言模型时可能会产生幻觉或错误信息。虽然这可能不符合许多客户支持场景的标准，但聊天机器人仍然可以帮助操作员起草回复，并在发送给用户之前进行核实。

接下来，我们将深入探讨如何使用 `GPT-3` 管理对话，并举例说明此工作流程的有效性。

首先，确保 `OPENAI_API_KEY` 和 `ACTIVELOOP_TOKEN` 环境变量已配置好您各自的 API 密钥和令牌。

由于我们将使用依赖于 `unstructured` 和 `selenium` Python 库的 `SeleniumURLLoader` LangChain 类，让我们通过 `pip` 安装它。建议安装最新版本的库。但请注意，代码已在版本 `0.7.7` 上进行了专门测试。

```
pip install unstructured==0.14.9 selenium==4.22.0
```

通过运行以下命令确保您已安装所需的包：`pip install langchain==0.0.208 deeplake openai==0.27.8 tiktoken`。现在，让我们继续导入必要的库。

```
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import SeleniumURLLoader
from langchain import PromptTemplate
```

这些库提供了管理 OpenAI 嵌入、处理向量存储、文本分割以及与 OpenAI API 交互的功能。它们有助于开发一个结合了检索和文本生成功能的、具有上下文感知能力的问答系统。我们聊天机器人的数据库将主要包含与技术问题相关的文章。

```
urls = ['', '']
```

#### 步骤 1：文档分割与嵌入计算

我们从提供的 URL 中检索文档，并使用 `CharacterTextSplitter` 将其分割成块，块大小为 `1000`，且无重叠。

```
loader = SeleniumURLLoader(urls=urls)
docs_not_splitted = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs_not_splitted)
```

随后，我们使用 `OpenAIEmbeddings` 计算嵌入向量，并将其保存到云端的 `Deep Lake` 向量存储中。在理想的生产环境中，我们可以将整个网站或课程内容上传到 `Deep Lake` 数据集，从而实现对数千甚至数百万文档的搜索。利用基于云的无服务器 `Deep Lake` 数据集，可以确保从不同位置运行的应用程序能够轻松访问同一个集中式数据集，而无需在专用机器上部署向量存储。

在运行以下代码之前，请确保您的 OpenAI 密钥已存储在 `OPENAI_API_KEY` 环境变量中。

```
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
my_activeloop_org_id = "Your Username"
my_activeloop_dataset_name = "langchain_course_customer_support"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)
# add documents to our Deep Lake dataset
db.add_documents(docs)
```

### 步骤 2：利用推荐技巧为 GPT-3 构建提示词

我们将构建一个提示词模板，该模板整合了角色提示、相关知识库数据以及用户的查询。

```
template = """You are an exceptional customer support chatbot that gently answer questions.
You know the following context information.
{chunks_formatted}
Answer to the following question from a customer. Use only information from the previous context information. Do not invent stuff.
Question: {query}
Answer:"""
prompt = PromptTemplate(
input_variables=["chunks_formatted", "query"],
template=template,
)
```

该框架将聊天机器人的角色设定为一位出色的客户支持助手。它接受两个输入变量：`chunks_formatted`（包含来自文章的预格式化段落）和 `query`（代表客户的询问）。目标是仅使用提供的段落生成准确的回答，避免生成虚假或捏造的信息。



### 第三步：使用温度为 0 的 GPT-3 模型进行文本生成

为了生成回复，我们首先检索与用户查询最相似的前 k 个文本片段。然后格式化提示词，并将其提交给温度为 0 的 GPT-3 模型。

```python
# 用户问题
query = "你的查询"
# 检索相关文本块
docs = db.similarity_search(query)
retrieved_chunks = [doc.page_content for doc in docs]
# 格式化提示词
chunks_formatted = "\n\n".join(retrieved_chunks)
prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)
# 生成答案
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
answer = llm(prompt_formatted)
print(answer)
```

**整个应用程序的代码：**

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import SeleniumURLLoader
from langchain import PromptTemplate
urls = ['', '']
# 使用 selenium 爬虫加载文档
loader = SeleniumURLLoader(urls=urls)
docs_not_splitted = loader.load()
# 我们将文档分割成更小的块
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs_not_splitted)
# 在执行以下代码之前，请确保
# 你的 OpenAI 密钥已保存在 "OPENAI_API_KEY" 环境变量中。
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# 创建 Deep Lake 数据集
# TODO: 在此处使用你的组织 ID。（默认情况下，组织 ID 是你的用户名）
my_activeloop_org_id = "你的用户名"
my_activeloop_dataset_name = "customer_support"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)
# 将文档添加到我们的 Deep Lake 数据集中
db.add_documents(docs)
# 让我们为客服聊天机器人编写一个提示词，该机器人使用从数据库中提取的信息来回答问题
template = """你是一位经验丰富的客服聊天机器人，能够全面回答问题。
你了解以下上下文信息。
{chunks_formatted}
回答用户的以下问题。仅使用之前上下文信息中的内容。不要发挥创意。
问题: {query}
答案:"""
prompt = PromptTemplate(
input_variables=["chunks_formatted", "query"],
template=template,
)
# 完整流程
# 用户问题
query = "你的问题？"
# 检索相关文本块
docs = db.similarity_search(query)
retrieved_chunks = [doc.page_content for doc in docs]
# 格式化提示词
chunks_formatted = "\n\n".join(retrieved_chunks)
prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)
# 生成答案
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
answer = llm(prompt_formatted)
print(answer)
```

**示例输出：**

```
Deep Lake Dataset in hub://didogrigorov/test_vector_db already exists, loading from the storage
Creating 38 embeddings in 1 batches of size 38:: 100%|██████████| 1/1 [00:11<00:00, 11.84s/it]
Dataset(path='hub://didogrigorov/test_vector_db', tensors=['embedding', 'id', 'metadata', 'text'])
tensor      htype      shape      dtype  compression
-------    -------    -------    -------  -------
embedding  embedding  (58, 1536)  float32   None
id        text      (58, 1)      str     None
metadata     json      (58, 1)      str     None
text       text      (58, 1)      str     None
AI-powered scams are fraudulent activities that use artificial intelligence technology to make them more convincing, easier, or cheaper to execute. These scams can take various forms, such as voice cloning of family and friends, fake identities and verification fraud, and even fake media generated by AI. It is important to be vigilant and cautious when receiving emails or messages from unknown sources, and to never click on suspicious links or open attachments unless you are 100% sure of their authenticity.
```

## 基础提示工程——所有应用中的共通之处

提示工程是与大型语言模型（LLM）交互并引导其产生期望输出的核心。以下是对基础提示工程的探索，以及一些有用的模板，助你开启旅程。

### 理解提示工程

提示工程涉及向 LLM 提供指令或提示，告知其所需完成的任务。这可以是一个简单的查询、一个详尽的任务描述，甚至是一个创意激发。指令的清晰、简洁和具体对于获得最佳结果至关重要。

**有效提示的好处**

- **更高的准确性：** 清晰的提示使 LLM 能够更好地理解意图，从而产生更相关、更精确的输出。
- **激发创造力：** 通过提供具体细节或约束，你可以引导 LLM 生成创新且独特的文本内容。
- **提升效率：** 精心设计的提示词能简化流程，确保 LLM 迅速聚焦于预期任务，节省时间和精力。

### 基础提示技巧

1.  **从简单开始：** 从直接的提示入手，例如问题或明确的指令。
2.  **精确是关键：** 融入关键词、示例和相关细节，有效引导 LLM 达到期望结果。
3.  **考虑语气和风格：** 指定期望的语气（正式、非正式、幽默）和风格（诗歌、剧本、邮件），以获得定制化结果。
4.  **利用参考：** 提供示例或参考资料，让 LLM 更清晰地理解你的期望。
5.  **实验与迭代：** 勇于尝试，根据输出结果不断优化提示词，以提升性能。

#### 提示模板示例

**信息检索**

- **问答：** "西班牙的首都是什么？"
- **摘要：** "总结以下研究论文的关键要点。"
- **主题探索：** "提供关于……的有趣事实。"

**创意写作**

- **故事开头：** "很久很久以前……"
- **诗歌生成器：** "以……的风格，创作一首关于……主题的诗歌。"
- **剧本写作：** "创建一个电影场景，展示两个角色之间的浪漫对话。"

**代码生成**

- **函数创建：** "编写一个 Python 函数，接收两个数字作为输入并返回它们的和。"
- **代码翻译：** "将此 C++ 代码翻译成 Python。"
- **错误修复：** "修复以下代码片段中的语法错误。"

**翻译**

- **语言翻译：** "将这句话从英语翻译成德语。"
- **方言转换：** "将此文本改写为美式英语。"
- **正式/非正式转换：** "为这封邮件写一个更正式的版本。"
- **语法错误：** "这段文本语法正确吗？"



