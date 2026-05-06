# 步骤 3：使用 `LLMChain` 类手动构建链

`chain = LLMChain(llm=llm, prompt=prompt)`

在此示例中，您正在使用 `llm_chain` 模板构建一个 LCEL 链。您提供了一个 OpenAI 语言模型（`llm`）和一个提示模板（`prompt`）作为参数，以自定义链的行为。

### 自定义 LCEL 链

构建好 LCEL 链后，下一步就是对其进行自定义。LCEL 链易于自定义是其最大的优势之一。您可以定制链的每个方面，以满足您的特定需求。

让我们探索几种自定义 LCEL 链的方法：

- **添加自定义逻辑**：您可以通过定义自己的函数或类，并将其包含在链的工作流程中，在 LCEL 链的步骤之间添加自定义逻辑。这允许您执行特定的数据转换或应用业务规则。
- **集成外部服务**：您可以使用工具与 API、数据库或第三方库进行交互。您只需定义必要的参数和身份验证详细信息，LangChain 将处理其余部分。
- **修改提示模板**：LCEL 链允许您使用变量、条件逻辑轻松定义和修改提示模板，甚至可以根据输入数据生成动态提示。

以下是自定义 LCEL 链的示例：

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import load_chain
from langchain.tools import APITool

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
api_tool = APITool(api_url="https://example.com/api")
chain = load_chain("llm_chain", llm=llm, prompt=prompt, tools=[api_tool])
```

在此示例中，我们通过添加一个 `APITool` 来与外部 API 交互，从而自定义 LCEL 链。我们定义了 `api_url` 参数来指定 API 端点，并在构建链时将工具包含在 `tools` 列表中。

### 执行 LCEL 链

构建并自定义好 LCEL 链后，就该让它运行起来了。您可以使用各种执行模式来适应不同的场景并优化性能。下面让我们探讨三种主要的执行模式。

### 流式执行

处理大型数据集时，您无需等待整个数据集加载到内存中，而是可以在数据到达时进行处理。您可以处理实时数据流，也可以分块处理数据。

要启用流式执行，您可以在调用链时使用 `streaming=True` 参数。操作如下：

`chain.run({"product": "smartphone"}, streaming=True)`

### 异步执行

借助异步执行，您可以同时运行多个任务，充分利用系统资源。这样，您可以并行执行多个独立的任务。

要使用异步执行，您可以利用 LCEL 链提供的 `arun` 方法。示例如下：

```python
import asyncio

async def generate_names(product):
    return await chain.arun({"product": product})

product_names = asyncio.run(generate_names("smartphone"))
```

### 批量执行

您可以使用批量执行在单次调用链时处理多个输入。这可以通过减少多次单独调用的开销来显著提高性能。

要执行批量执行，您需要向链的 `apply` 方法传递一个输入列表。示例如下：

```python
products = ["smartphone", "laptop", "smartwatch"]
product_names = chain.apply([{"product": product} for product in products])
```

## LCEL 链的可观测性

在构建和调试 LCEL 链时，可观测性至关重要。您可以深入了解链的内部工作原理，跟踪数据流，并识别潜在的瓶颈或问题。

LCEL 链提供了内置的可观测性功能，例如日志记录和追踪。您可以访问链中每个步骤的详细日志和追踪信息，例如输入和输出数据、执行时间以及可能发生的任何错误或异常。

要在 LCEL 链中启用可观测性，您可以在构建链时使用 `verbose=True` 参数。示例如下：

`chain = load_chain("llm_chain", llm=llm, prompt=prompt, verbose=True)`

使用 `RunnableSequence` 时，您的代码可能如下所示：

```python
from langchain.globals import set_debug
set_debug(True)
```

一旦启用了可观测性，您就可以轻松监控链的执行情况，并获得有价值的见解，用于调试和优化。

### LCEL 链的类型

让我们讨论一下 LangChain 文档中提到的几种 LCEL 链类型，我已将其整理成表格。

| 链 | 使用时机 | 构造函数 |
|-------|-------------|-------------|
| `create_stuff_documents_chain` | 当您想将文档列表格式化为提示并传递给 LLM 时，此链是您的首选。只需确保文档适合您所使用的 LLM 的上下文窗口即可。 | `create_stuff_documents_chain` |
| `create_openai_fn_runnable` | 如果您想使用 OpenAI 函数调用来可选地结构化输出响应，这是您的首选链。您可以传入多个函数，但不要求全部调用。 | `create_openai_fn_runnable` |
| `create_structured_output_runnable` | 当您想使用 OpenAI 函数调用来强制 LLM 使用特定函数进行响应时，这是您的首选链。您只能传入一个函数，并且该链将始终返回此响应。当您每次都需要结构化输出时，这非常有用。 | `create_structured_output_runnable` |
| `load_query_constructor_runnable` | 此链用于生成查询。您指定一个允许的操作列表，它会返回一个可运行对象，该对象将自然语言查询转换为这些操作。 | `load_query_constructor_runnable` |
| `create_sql_query_chain` | 当您想根据自然语言为 SQL 数据库构建查询时，可以使用此链。无需再与复杂的 SQL 语法作斗争！ | `create_sql_query_chain` |
| `create_history_aware_retriever` | 此链接收对话历史记录，并使用它来生成搜索查询，然后将该查询传递给底层的检索器。 | `create_history_aware_retriever` |
| `create_retrieval_chain` | 当您想根据用户查询检索相关文档时，此链是您的首选。它接收用户的输入，将其传递给检索器以获取相关文档，然后将这些文档与原始输入结合起来，使用 LLM 生成响应。 | `create_retrieval_chain` |

## 使用查询构造函数链生成命令

现在，让我们看一个如何在代码中使用 `load_query_constructor_runnable` 链的示例：

```python
from langchain.chains import load_query_constructor_runnable

# 定义允许的操作
allowed_operations = ["search", "filter", "sort"]

# 加载查询构造函数链
query_constructor = load_query_constructor_runnable(allowed_operations)
```



