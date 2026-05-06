# 索引

在本节中，让我们讨论索引在管理文档方面的强大功能。LangChain 索引 API 允许你加载文档并使其与向量存储保持同步，同时避免重复内容、重写未更改的文档以及不必要地重新计算嵌入。这一切都是为了节省你的时间和金钱，同时改进你的向量搜索结果。

在底层，索引 API 使用一个 `RecordManager`，它通过计算每个文档的哈希值并存储一些关键信息（如下所示）来跟踪对向量存储的文档写入操作：

- 文档哈希（页面内容和元数据的唯一指纹）
- 写入时间（文档添加或更新的时间）
- 源 ID（一种标识文档原始来源的方式）

你还可以使用索引 API 提供的删除模式来控制索引新文档时如何处理现有文档。

你可以从以下模式中选择：

1. **无**：此模式让你可以自由地手动管理内容，因为 API 不会执行任何自动清理。
2. **增量**：如果源文档或派生文档发生更改，此模式会持续清理内容的先前版本。
3. **完整**：此模式在索引过程结束时执行彻底清理，删除当前索引批次中不再存在的任何文档。

这些删除模式有助于确保你的向量存储保持精简。

## 关键要点

在本章中，你探索了检索增强生成（RAG），这是一种通过检索引入额外上下文来增强大型语言模型（LLM）的强大技术。

以下是我们的学习内容。

**理解 RAG**：我们定义了 RAG，并理解了它在增强 LLM 方面的重要性，尤其是在需要最新或特定领域知识的情况下。我们看到了 RAG 如何减少幻觉、进行事实核查、提供特定领域知识以及增强 LLM 的响应。

**RAG 架构**：我们研究了 RAG 的两个主要组成部分——索引与检索和生成。索引组件涉及加载、拆分和存储数据，而检索和生成组件则涉及检索相关数据并生成答案。

**实现 RAG**：我们探索了使用 LangChain、Pinecone 和 OpenAI 实现 RAG 的实践方法。我们学习了如何设置知识库、实现检索以及使用 RAG 开发问答应用程序。

## 复习题

1. 检索增强生成（RAG）的主要目的是什么？
   - A. 提高 LLM 响应的速度
   - B. 将基于检索的方法与大型语言模型相结合，以提高生成响应的准确性和相关性
   - C. 减小 LLM 模型的规模
   - D. 增加 LLM 模型的训练时间

2. 以下哪些组件是 RAG 架构的一部分？
   - A. 索引与检索和生成
   - B. 分词与归一化
   - C. 加密与解密
   - D. 缓存与日志记录

3. RAG 过程中的嵌入阶段涉及什么？
   - A. 清理文本并删除无关信息
   - B. 将转换后的数据转换为称为嵌入的数值表示
   - C. 从源获取数据并将其格式化为可处理的形式
   - D. 将嵌入存储在数据库中

4. 哪个 LangChain 组件负责将文本拆分成更小的块？
   - A. 文档加载器
   - B. 文本分割器
   - C. 向量存储
   - D. 检索器

5. 使用 `CacheBackedEmbeddings` 类的目的是什么？
   - A. 将嵌入存储在远程服务器上
   - B. 缓存嵌入以避免重新计算它们
   - C. 将文本转换为嵌入
   - D. 存储文档元数据

6. 哪个向量存储被提及支持异步操作？
   - A. Chroma
   - B. Milvus
   - C. Qdrant
   - D. Pinecone

7. `Self-Query Retriever` 使用什么来转换用户输入？
   - A. 正则表达式
   - B. 机器学习模型
   - C. 预定义的关键词集
   - D. 语言模型

8. 在 LangChain 的上下文中，什么是嵌入？
   - A. 一段文本数据
   - B. 文本的数值表示
   - C. 文档的存储格式
   - D. 一种拆分文本的方法

9. 以下哪项不是 LangChain 索引 API 中的删除模式？
   - A. 无
   - B. 增量
   - C. 完整
   - D. 部分

10. 在信息检索系统中使用文本嵌入的好处是什么？
    - A. 它们增加了数据库的存储容量。
    - B. 它们捕获文本的语义含义，以便进行比较和检索。
    - C. 它们降低了文本处理的复杂性。
    - D. 它们简化了分词过程。

## 答案

1. B. 将基于检索的方法与大型语言模型相结合，以提高生成响应的准确性和相关性。
2. A. 索引与检索和生成
3. B. 将转换后的数据转换为称为嵌入的数值表示
4. B. 文本分割器
5. B. 缓存嵌入以避免重新计算它们
6. C. Qdrant
7. D. 语言模型
8. B. 文本的数值表示
9. D. 部分
10. B. 它们捕获文本的语义含义，以便进行比较和检索。

## 术语表

本术语表提供了本章讨论的与检索增强生成（RAG）和信息检索系统相关的关键技术术语的定义。

**`CacheBackedEmbeddings`**：一个嵌入器的包装器，它将嵌入缓存到键值存储中，使用哈希后的文本作为键来高效地存储和检索嵌入

**`CharacterTextSplitter`**：一个用于根据指定字符或字符集将文本拆分成更小块的类，有助于维护上下文并确保块大小可控

**文档加载器**：一种用于从各种来源（如 HTML、PDF 或代码文件）加载文档，并将其转换为应用程序可处理格式的工具

**文档**：一个包含一段文本及其关联元数据的容器，用于信息检索系统中的处理和分析

**嵌入**：文本的一种数值表示，捕获其语义含义，允许基于相似性进行比较和检索

**FAISS（Facebook AI 相似性搜索）**：一个用于高效相似性搜索和稠密向量聚类的开源库，常用于实现向量存储

**索引**：组织数据以提高信息检索速度和效率的过程，通常通过创建允许快速搜索和更新的结构来实现

**`JSONLoader`**：一种用于加载 JSON 数据并根据 `jq` 模式提取特定信息的工具，适用于处理和分析 JSON 文件

**多向量检索器**：一种为每个文档创建多个向量的检索器，捕获内容的不同方面以提高检索准确性

**父文档检索器**：一种为每个文档索引多个块，并根据其嵌入找到最相似块，然后返回整个父文档以获得全面结果的检索器

**Qdrant**：一个支持异步操作的向量存储，用于信息检索系统中的高效相似性搜索和检索

**RAG（检索增强生成）**：一种将基于检索的方法与大型语言模型相结合的技术，通过检索引入额外上下文来提高生成响应的准确性和相关性

**`RecursiveCharacterTextSplitter`**：一种文本分割器，使用递归方法根据指定的分隔符将文本分成更小的块，确保均匀分布并保持逻辑结构

**检索器**：一种用于根据非结构化查询获取相关文档的工具，有助于在信息检索系统中快速访问信息

**自查询检索器**：一种使用语言模型将用户输入转换为查询字符串和元数据过滤器的检索器，适用于基于元数据而非文本相似性检索文档

**文本嵌入**：文本的向量形式数值表示，用于捕获语义含义并促进信息检索系统中的比较和检索

**文本分割器**：一种用于将文本分成更小块的工具，确保每个块大小可控并保持语义连贯性

**分词**：将文本分解为更小单元（如单词或短语）以供机器学习模型处理的过程

**向量存储**：一种以向量格式存储数据的数据库，允许基于向量比较进行高效的相似性搜索和检索

**向量存储检索器**：一种使用向量存储根据嵌入查找相似文档的检索器，适用于快速简便的检索任务

## 参考文献

这些参考文献为在 LangChain、Pinecone 和 OpenAI 技术背景下实现检索增强生成（RAG）系统提供了基础知识。

1. Lewis, P., 等人 (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." arXiv:2005.11401. 这篇研究论文介绍了 RAG 模型，并解释了其在知识密集型任务中的架构和性能。
2. LangChain 文档. "Question Answering Use Cases." [`python.langchain.com/v0.1/docs/use_cases/question_answering/`](https://python.langchain.com/v0.1/docs/use_cases/question_answering/) 这份官方文档是关于使用 LangChain 构建问答系统的详细指南。
3. Pinecone 文档. "Retrieval-Augmented Generation with Pinecone." [`www.pinecone.io/learn/retrieval-augmented-generation/`](https://www.pinecone.io/learn/retrieval-augmented-generation/) 这是 Pinecone 的文档，讨论了如何使用 Pinecone 的向量数据库进行 RAG。
4. Kumar, A. (2023). "Retrieval-Augmented Generation (RAG): From Theory to LangChain Implementation." [`towardsdatascience.com/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2`](https://towardsdatascience.com/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2) 这篇文章提供了使用 LangChain 实现 RAG 的实践概述和指南。
5. OpenAI 帮助中心. "Retrieval-Augmented Generation (RAG) and Semantic Search for GPTs." [`help.openai.com/en/articles/8868588-retrieval-augmented-generation-rag-and-semantic-search-for-gpts`](https://help.openai.com/en/articles/8868588-retrieval-augmented-generation-rag-and-semantic-search-for-gpts) 这是 OpenAI 关于将 RAG 应用于 OpenAI 模型的资源。

