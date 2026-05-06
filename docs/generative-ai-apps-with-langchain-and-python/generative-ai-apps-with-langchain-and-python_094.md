# 使用查询构造器将自然语言转换为允许的操作

`natural_language_query = "查找所有价格低于 100 美元的产品并按价格排序"`  
`structured_query = query_constructor(natural_language_query)`  
`print(structured_query)`

在此示例中，我们定义了一个允许的操作列表："搜索"、"筛选"和"排序"。然后加载`query_constructor_runnable`链，并传入允许的操作。该链接收自然语言查询，并将其转换为指定的允许操作。

您提供自然语言查询："查找所有价格低于 100 美元的产品并按价格排序"。`query_constructor`链处理此查询，并基于允许的操作返回结构化查询。

以下是您将获得的输出：

```
{
  "operations": [
    {
      "operation": "search",
      "query": "products"
    },
    {
      "operation": "filter",
      "condition": "price < 100"
    },
    {
      "operation": "sort",
      "key": "price"
    }
  ]
}
```

在上述示例中，自然语言查询被转换为包含特定操作（如"搜索"、"筛选"和"排序"）的结构化查询。这种结构化格式更易于您的应用程序处理和执行。

## 使用传统链进行构建

构建传统链是一个直接的过程。您只需创建所需链类的实例，并提供必要的参数。您需要查阅所用特定链的文档，因为每个链都有自己的一套必需参数和可选参数。

### 构建传统链

以下是构建传统链的示例：

```python
from langchain.chains import LLMChain
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
chain = LLMChain(llm=llm, param1=value1, param2=value2)
```

在此示例中，您导入所需的链类（`MyCustomChain`）并创建其实例。然后提供必要的参数，例如语言模型（`llm`）以及任何自定义参数（`param1`和`param2`）。

### 执行传统链

执行传统链与构建它们一样简单。大多数传统链都提供一个`run`方法，该方法接收输入数据并返回结果。以下是一个示例：

```python
result = chain.run(input_data)
print(result)
```

在此示例中，您只需在链实例上调用`run`方法，并传入`input_data`。链处理输入并返回结果，然后您打印该结果。

### 传统链的类型

现在您对传统链有了基本了解，让我们探索一些最常用的链。

| 链 | 使用场景 |
|-------|-------------|
| `APIChain` | 当您想将查询转换为 API 请求、执行该请求，然后使用 LLM 解释响应时。 |
| `OpenAPIEndpointChain` | 类似于`APIChain`，但针对 OpenAPI 端点进行了优化。 |
| `ConversationalRetrievalChain` | 用于与文档进行对话，利用之前的对话历史来优化查询。 |
| `StuffDocumentsChain` | 当您有一份文档列表，且这些文档能放入 LLM 的上下文窗口时。 |
| `ReduceDocumentsChain` | 用于通过迭代缩减的方式并行处理大量文档。 |
| `MapReduceDocumentsChain` | 类似于`ReduceDocumentsChain`，但在缩减文档之前会先进行一次 LLM 调用。 |
| `RefineDocumentsChain` | 用于基于多个文档顺序优化答案。 |
| `MapRerankDocumentsChain` | 当您想基于置信度最高的单个文档来回答问题时。 |
| `ConstitutionalChain` | 用于在链的答案中强制执行宪法原则。 |
| `LLMChain` | 用于与 LLM 交互的基本链。 |
| `ElasticsearchDatabaseChain` | 用于对 Elasticsearch 数据库提出自然语言问题。 |
| `FlareChain` | 一种用于探索性目的的高级检索技术。 |
| `GraphCypherQAChain` | 用于从自然语言构建 Cypher 查询并针对图数据库执行。 |
| `LLMMath` | 用于将用户问题转换为数学问题并执行。 |
| `LLMCheckerChain` | 使用第二次 LLM 调用来验证初始答案。 |
| `LLMSummarizationChecker` | 通过多次 LLM 调用创建摘要以提高准确性。 |

这些只是可用传统链中的几个示例。每个链都有其独特的优势和用例，使您能够处理广泛的任务。

## 使用传统链构建实际应用

您可以使用传统链构建非常有效的实际生成式 AI 应用。以下是一些流行的示例。

### 使用`ConversationalRetrievalChain`的文档聊天机器人应用

假设您想使用`ConversationalRetrievalChain`与文档进行对话。以下是一个简化示例：

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever

# 初始化检索器和 LLM
retriever = ContextualCompressionRetriever(...)
llm = ChatOpenAI(temperature=0)

# 创建 ConversationalRetrievalChain
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
```



