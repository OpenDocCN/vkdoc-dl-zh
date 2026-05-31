# 定义要嵌入的客户评论列表

`reviews = [`

`"我非常喜欢这个产品！它让我的生活变得轻松多了，我无法想象回到以前的样子。",`

`"起初我有点怀疑，但使用这项服务几周后，我印象非常深刻。客户支持是一流的，功能也正好是我需要的。",`

`"我使用这个软件已经有一段时间了，它确实提高了我的工作效率。用户界面很直观，与其他工具的集成也很无缝。",`

`"我与这家公司合作非常愉快。他们按时交付了项目，质量超出了我的预期。我肯定会再次与他们合作。",`

`"我对这个产品并不完全满意。虽然它有一些有用的功能，但我发现它在某些方面有所欠缺，而且价格对于所获得的东西来说似乎有点高。"`

`]`

你使用 `embeddings_model` 的 `embed_documents` 方法来嵌入评论列表。生成的嵌入向量存储在 `embeddings` 变量中：

```
# 使用 embed_documents 方法嵌入评论列表
embeddings = embeddings_model.embed_documents(reviews)
```

## 第 7 章 使用检索增强生成（RAG）构建高级问答与搜索应用

你打印嵌入向量的数量（应与评论数量相等）以及每个嵌入向量的长度：

```
# 打印嵌入向量的数量和每个嵌入向量的长度
print(f"嵌入向量数量: {len(embeddings)}")
print(f"每个嵌入向量的长度: {len(embeddings[0])}")
```

你定义一个名为 `query_text` 的查询文本，询问客户评论中提到的积极方面：

```
# 定义要嵌入的查询文本
query_text = "客户评论中提到了哪些积极方面？"
```

你使用 `embeddings_model` 的 `embed_query` 方法来嵌入查询文本。生成的嵌入向量存储在 `embedded_query` 变量中：

```
# 使用 embed_query 方法嵌入查询文本
embedded_query = embeddings_model.embed_query(query_text)
```

你打印嵌入查询向量的长度及其前五个元素：

```
# 打印嵌入查询向量的长度
print(f"嵌入查询向量的长度: {len(embedded_query)}")
# 打印嵌入查询向量的前 5 个元素
print(f"嵌入查询向量的前 5 个元素: {embedded_query[:5]}")
```

## 第 7 章 使用检索增强生成（RAG）构建高级问答与搜索应用

你可以嵌入客户评论，并使用查询文本来执行各种分析和任务，例如：

- 识别评论中提到的共同主题或话题
- 根据嵌入向量将相似的评论聚类在一起
- 搜索与特定查询或主题最相关的评论
- 进行情感分析，以确定评论中表达的总体情感
- 比较不同评论的嵌入向量，以发现相似之处或差异

在后续章节中，我们将看到如何使用这些查询将其存储在向量存储中，然后获取这些查询的结果。

### 缓存嵌入向量

让我们探讨一下缓存嵌入向量的话题，这很重要，因为它可以节省你的时间、金钱和计算资源。

在嵌入向量的上下文中，缓存意味着存储或临时保存计算出的嵌入向量，以避免每次需要时都重新计算它们。

你可以使用 `CacheBackedEmbeddings` 类进行缓存。可以把它看作是一个包裹在你的嵌入器周围的包装器。它使用哈希后的文本作为键，将嵌入向量缓存在键值存储中，这可以高效地存储和检索嵌入向量。

你应该首先创建一个 `CacheBackedEmbeddings` 实例，并调用 `from_bytes_store` 方法。你需要提供一些重要的信息，例如：

## 第 7 章 使用检索增强生成（RAG）构建高级问答与搜索应用

1. `underlying_embedder`：这是你用于嵌入文本的嵌入器。
2. `document_embedding_cache`：你需要一个 `ByteStore` 来存储缓存的文档嵌入向量。可以把它想象成一个安全的保险库，你的嵌入向量将在其中安然无恙。
3. `batch_size`（可选）：如果你想批量嵌入多个文档，可以指定在存储更新之间要嵌入的文档数量。这就像告诉你的缓存机器一次要处理多少项。
4. `namespace`（可选）：为了避免与其他缓存冲突，你可以为你的文档缓存提供一个命名空间。这是一个唯一的名称标签，用于避免缓存之间相互混淆。

现在，这里有一个重要的提示：如果你使用不同的嵌入模型，请确保设置 `namespace` 参数以避免任何冲突。你肯定不希望你的缓存混淆，开始混合来自不同模型的嵌入向量！

#### 缓存嵌入向量的代码演练

让我们看一个如何使用 `CacheBackedEmbeddings` 与向量存储的示例。你将使用本地文件系统存储嵌入向量，并使用 FAISS 向量存储进行检索。

以下是完整的、端到端的工作代码。

## 第 7 章 使用检索增强生成（RAG）构建高级问答与搜索应用

首先，你必须安装或升级 `langchain-openai` 和 `faiss-cpu` 包，这是代码运行所必需的：

```
# 安装必要的包
!pip install --upgrade --quiet langchain-openai faiss-cpu
```

你从 `langchain.storage` 模块导入 `LocalFileStore` 类，用于在本地存储嵌入向量：

```
# 导入所需模块
from langchain.storage import LocalFileStore
```

然后，你从 `langchain.document_loaders` 模块导入 `TextLoader` 类，用于加载文本文档：

```
from langchain.document_loaders import TextLoader
```

你从 `langchain.vectorstores` 模块导入 `FAISS` 类，以使用 FAISS 库实现向量存储：

```
from langchain.vectorstores import FAISS
```

### 为什么选择 FAISS-CPU？

FAISS（Facebook AI 相似度搜索）是 Facebook AI Research 开发的一个库，用于高效的高维向量相似度搜索和聚类。让我们看看为什么 `faiss-cpu` 在我们的上下文中至关重要：

**高效的向量存储与检索**：FAISS 旨在高效处理大规模向量搜索。它提供了用于存储和检索高维向量的优化算法，使其成为涉及大型数据集的应用程序的理想选择。

## 第 7 章 使用检索增强生成（RAG）构建高级问答与搜索应用

**可扩展性**：FAISS 可以扩展到数百万甚至数十亿个向量，确保即使数据集增长，你的应用程序也能保持高性能。

**多功能性**：FAISS 支持多种索引策略，因此你可以为你的特定用例选择最佳策略。例如，你可以在快速近似最近邻搜索和精确搜索之间进行选择。

**基于 CPU**：`faiss-cpu` 包针对 CPU 使用进行了优化，因此对于 GPU 资源有限或不可用的环境来说，它是一个很好的替代方案。这确保了你可以将应用程序部署并在各种硬件配置上运行。

如你所见，这些优势可以帮助你构建一个强大的向量搜索系统。

下面，你从 `langchain_openai` 模块导入 `OpenAIEmbeddings` 类，用于使用 OpenAI 的 API 生成嵌入向量：

```
from langchain_openai import OpenAIEmbeddings
```

然后从 `langchain.text_splitter` 模块导入 `CharacterTextSplitter` 类，用于将文本分割成块：

```
from langchain.text_splitter import CharacterTextSplitter
```

最后，你必须从 `langchain.embeddings` 模块导入 `CacheBackedEmbeddings` 类，用于缓存嵌入向量：

```
from langchain.embeddings import CacheBackedEmbeddings
```



#### 设置 OpenAI API 密钥

```python
import os

os.environ["OPENAI_API_KEY"] = "Your OpenAI key"
```

