# 第 6 章：使用链构建智能聊天机器人和自动化分析系统

## 使用 `LLMChain` 构建文本生成应用

你可以使用 `LLMChain` 将语言模型（LLM）与提示模板结合起来生成文本。以下是一个快速示例：

```python
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

template = "What is a good name for a company that makes {product}?"
prompt = PromptTemplate(template=template, input_variables=["product"])
llm = OpenAI(temperature=0.9)
chain = LLMChain(llm=llm, prompt=prompt)
company_name = chain.run("colorful socks")
print(company_name)
```

在此示例中，你首先定义了一个提示模板和一个 OpenAI 语言模型。然后，通过组合 `llm` 和 `prompt` 创建了一个 `LLMChain`，并使用产品输入运行该链，最后打印生成的公司名称。

## 使用 `ConversationChain` 构建对话应用

当你需要与语言模型进行对话并跟踪对话历史时，你将使用此链。以下是一个示例：

```python
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
conversation = ConversationChain(llm=llm)
output = conversation.predict(input="Hi, how are you?")
print(output)
output = conversation.predict(input="What's the weather like today?")
print(output)
```

在此示例中，你使用一个 OpenAI 语言模型创建了一个 `ConversationChain`，然后使用 `predict` 方法与 AI 助手进行对话。该链会跟踪对话历史，并维持更自然、更具上下文关联的交互。

## 使用 `RetrievalQA` 构建问答应用

你将使用 `RetrievalQA` 链对文档集合执行问答。以下是一个示例：

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai.embeddings import OpenAIEmbeddings

loader = TextLoader("path/to/documents.txt")
index = VectorstoreIndexCreator().from_loaders([loader])
embedding = OpenAIEmbeddings(openai_api_key=MY_OPENAI_API_KEY)
llm = OpenAI(temperature=0.9)
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index.vectorstore.as_retriever())
query = "What is the capital of France?"
result = chain.run(query)
print(result)
```

在此示例中，你使用 `TextLoader` 加载了一个文档集合，然后使用 `VectorstoreIndexCreator` 创建了一个索引。接下来，你使用一个 OpenAI 语言模型实例化了一个 `RetrievalQA` 链，并将检索策略指定为 "stuff"。最后，你使用一个查询运行该链并打印结果。

## 使用 `MapReduceChain` 的文档处理应用

`MapReduceChain` 通过将文档分割成块、对每个块应用映射函数，然后归约结果，帮助你并行处理文档。以下是一个示例：

```python
from langchain.chains import MapReduceChain
from langchain.chains.mapreduce import combine_results
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader

loader = TextLoader("path/to/documents.txt")
documents = loader.load()
llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(template="Summarize this text: {text}", input_variables=["text"])
chain = MapReduceChain.from_params(
    map_prompt=prompt,
    combine_prompt=prompt,
    llm=llm,
    chunk_size=1000,
    reduce_chunk_overlap=0,
)
result = chain.run(documents)
print(result)
```

在此示例中，你使用 `TextLoader` 加载了一个文档集合，然后通过指定映射和合并提示、语言模型以及块大小创建了一个 `MapReduceChain`。最后，你在文档上运行该链并打印汇总结果。

### 教育案例研究：使用 MapReduce 链分析教育数据

某教育研究机构使用 MapReduce 链处理大量学术数据，用于分析和报告。

**实施**：MapReduce 链将数据集分割成可管理的块，对每个块应用分析模型，然后汇总结果，生成关于学生表现和学习趋势的综合报告。

**成果**：高效的处理流程缩短了研究论文的周转时间，并提供了对教育效果的更深入洞察，支持对教学策略进行有针对性的改进。

## 使用链组合策略构建更复杂的工作流应用

我们已经了解了不同类型的链以及如何单独使用它们。在本节中，我们将探讨如何组合链以应对最复杂的工作流。

链组合的核心是通过连接多个链来创建工作流。你需要先将复杂任务分解为更小、更易管理的步骤，并将每个步骤分配给特定的链。

让我们探索一些可用于组合链的策略，以构建复杂的真实世界生成式 AI 应用。

### 使用顺序链的数据汇总应用

当一个链的输出成为下一个链的输入，以创建具有线性数据处理流程的应用时，你将使用顺序链。

以下是一个顺序链工作流的示例：

```python
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# 链 1：根据主题生成关键词
topic_prompt_template = "Generate relevant keywords for the topic: {topic}"
topic_prompt = PromptTemplate(template=topic_prompt_template, input_variables=["topic"])
topic_chain = LLMChain(llm=OpenAI(), prompt=topic_prompt)

# 链 2：根据关键词获取相关数据
data_prompt_template = "Fetch data related to the following keywords: {keywords}"
data_prompt = PromptTemplate(template=data_prompt_template, input_variables=["keywords"])
data_chain = LLMChain(llm=OpenAI(), prompt=data_prompt)

# 链 3：汇总获取的数据
summary_prompt_template = "Summarize the following data: {data}"
summary_prompt = PromptTemplate(template=summary_prompt_template, input_variables=["data"])
summary_chain = LLMChain(llm=OpenAI(), prompt=summary_prompt)

# 按顺序组合链
sequential_chain = SequentialChain(chains=[topic_chain, data_chain, summary_chain], input_variables=["topic"], output_variables=["summary"])
```