# 使用 LangChain 进行检索任务的示例

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# 加载并准备数据
loader = TextLoader('your_data.txt')
documents = loader.load()

# 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# 初始化语言模型
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# 创建检索链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 使用链
query = "你的问题"
result = qa_chain.invoke({"query": query})
print(result['result'])
```

## 第 3 章 构建问答与聊天机器人应用

这个简化示例展示了如何使用 LangChain 创建一个基于检索的基本问答系统。

1. 首先，从文本文件中加载文档。
2. 然后创建一个向量存储（`FAISS`），用于高效存储和检索文档嵌入。
3. 初始化一个语言模型（`ChatOpenAI`）。
4. 创建一个 `RetrievalQA` 链，将检索器与语言模型结合起来。
5. 最后，使用该链基于加载的文档回答问题。

通过这种设置，你可以询问关于文档内容的问题，系统将检索相关信息并生成答案。

请记得将 `your_data.txt` 替换为实际数据文件的路径，并确保已为 OpenAI 设置好必要的 API 密钥。

## 生态系统与集成

你应该意识到，LangChain 生态系统不仅仅包含这些软件包。它还为你提供了多种工具和集成，可用于部署、监控和管理你的应用程序：

### 第 3 章 构建问答与聊天机器人应用

**LangTemplates**：这些是即用型模板，可帮助你快速上手常见用例。`LangTemplates` 为诸如回复邮件、分析文本等用例提供了坚实的起点。

**LangServe**：你将使用 `LangServe` 将 LangChain 应用程序部署为 REST API。你可以轻松地将链转换为广泛可访问的 Web 服务。

**LangSmith**：`LangSmith` 提供与 LangChain 无缝集成的调试和监控工具。你可以利用它们获取分析洞察，并用于进一步优化你的生成式 AI 应用。

通过将软件包拆分为核心组件和社区组件，LangChain 保持了轻量级和灵活的结构，使你能够为各种用例构建应用程序。

## 使用 LangChain 模型与 LLM

在本节中，我们将探讨 LangChain 中的模型、其输入和输出。

### 模型 IO：LangChain 的核心功能

任何语言模型应用的核心都是模型本身。模型 IO 是一种机制，通过它你可以无缝地与大型语言模型（LLM）进行通信。你将使用它来连接这些卓越 LLM 模型的庞大知识和能力。

### 第 3 章 构建问答与聊天机器人应用

LangChain 模型 IO 的优势之一在于，它允许你轻松地在不同提供商之间切换。这是一个巨大的优势，因为即使你最初使用 OpenAI 启动项目，也可以选择探索来自 Google Cloud 或 Anthropic 的其他模型的能力。你可以毫不费力地进行切换，而无需重写整个代码库。

### 使用 LangChain 的大型语言模型（LLM）

在本节中，我们将讨论不同类型的语言模型，并探讨 LangChain 如何简化与它们的交互。

#### LangChain 模型的类型

你可以在 LangChain 中使用两种类型的模型，即通用 LLM 和聊天模型：

**LLM 模型**：通用 LLM 模型也称为文本补全模型，它就像一个文字魔法师，能够预测输入文本最可能的后续内容。你可以发送一个字符串提示作为输入，并生成一个字符串补全作为输出。这是一种直接的方法，就像拥有一个智能文本生成器。想象一下，你以“一石二”开头，模型会将其补全为“鸟”。

**聊天模型**：聊天模型建立在 LLM 之上，但专门针对使用消息进行来回对话进行了调优。你发送一条人类消息，AI 用自己的消息回复，对话就这样不断来回进行。这就像有一个随时准备聊天的人！

### 第 3 章 构建问答与聊天机器人应用


好的，作为一名高级文档工程师和翻译员，我将严格遵循您提供的注意事项和示例，将给定的英文文本翻译成中文。


你也可以在聊天模型中使用系统级提示词，为 AI 模型赋予个性。这些系统级提示词能帮助你为 AI 设定特定的语气或角色，例如一位友好耐心的导师，AI 会据此进行对话。

在选择模型时，你需要记住一点：不同的模型有其自身的特点和偏好，你应该选择最适合你应用的模型。我们将在下一章更详细地讨论 LLM 模型。

**注意** 关于`LangChain`支持的 LLM 完整列表，请查阅`LangChain`文档：[`python.langchain.com/`](https://python.langchain.com/v0.2/docs/integrations/llms/)

[`python.langchain.com/v0.2/docs/integrations/llms/`](https://python.langchain.com/v0.2/docs/integrations/llms/)

## 构建一个简单的问答应用

好了，让我们动手写一些代码来更好地理解这一点。

### 第 1 步：导入库

```
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

首先，你必须从`LangChain`库中导入必要的类。`OpenAI`是 LLM，`ChatPromptTemplate`用于构建提示词，而`StrOutputParser`用于解析输出。

### 第 2 步：输出解析器

然后，你必须创建一个`StrOutputParser`的实例，它将把 LLM 的输出转换为字符串：

```
output_parser = StrOutputParser()
```

### 第 3 步：创建提示词模板

接着，你必须创建一个包含两条消息的聊天提示词模板：

-   一条为 AI 设定上下文（扮演健康专家）的系统消息
-   一条包含占位符`{input}`用于实际查询的用户消息

```
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world recognized wellness expert especially in cardio activities."),
    ("user", "{input}")
])
```

### 第 4 步：初始化 LLM

然后，你初始化 OpenAI 语言模型。记得将`"openai_api_key"`替换为你实际的 OpenAI API 密钥：

```
llm = OpenAI(api_key="openai_api_key")
```

### 第 5 步：创建链

在这里，你使用`|`运算符创建一个处理链。它按顺序连接了提示词模板、LLM 和输出解析器。我们将在关于链的章节中更详细地讨论链。

```
chain = prompt | llm | output_parser
```

### 第 6 步：调用链

最后，你使用特定的输入来调用这个链。该链将：

1.  使用给定的输入格式化提示词模板。
2.  将格式化后的提示词发送给 OpenAI LLM。
3.  使用`StrOutputParser`解析 LLM 的响应。

### 第 7 步：输出

最终的输出将是一个字符串，包含 LLM 关于每天步行一英里好处的回复，其表述方式就像来自一位专攻有氧运动的健康专家。

```
output = chain.invoke({"input": "What are the benefits of walking a mile a day?"})
```

恭喜你，你刚刚采用了一种简单而强大的方式，使用`LangChain`与 LLM 进行交互，并在类似链的处理流程中使用了结构化的输入和输出。

### 完整的端到端工作代码

以下是完整的端到端工作代码：

```
!pip install langchain_core==0.2.17 langchain_openai==0.1.16

from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world recognized wellness expert especially in cardio activities."),
    ("user", "{input}")
])

llm = OpenAI(api_key="your open AI key")

# 创建一个链，定义你的应用如何与 LLM 交互
chain = prompt | llm | output_parser

output = chain.invoke({"input": "What are the benefits of walking a mile a day?"})
print(output)
```

**注意** 运行此代码时，如果碰巧遇到错误，可以将你遇到的错误粘贴到`Gemini`或`ChatGPT`中检查，它们会根据你使用的`openai`和`LangChain`的不同版本，为你指出正确的语法。

## 构建一个对话式应用

一旦你掌握了窍门，与聊天模型聊天就非常简单了。首先，你需要从`langchain_openai`中导入`ChatOpenAI`类。这将使你能够与 OpenAI 的聊天模型（如 GPT-3.5 Turbo 和 GPT-4）进行交互。

具体操作如下。

### 第 1 步：导入必要的模块

首先，你必须导入与 OpenAI 聊天模型交互、格式化消息以及处理环境变量所需的模块：

```
# 导入必要的模块
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
```

### 第 2 步：设置 API 密钥

你必须将 OpenAI API 密钥设置为环境变量。这比在脚本中硬编码密钥更安全。记得将`"Your OpenAI Key"`替换为你实际的 API 密钥：

```
# 将 API 密钥设置为环境变量（更安全）
os.environ["OPENAI_API_KEY"] = "Your OpenAI Key"
```

### 第 3 步：创建 ChatOpenAI 对象

你必须创建一个`ChatOpenAI`类的实例，它代表基于聊天的模型。`model`参数指定要使用的具体模型，这里是`"gpt-3.5-turbo-0125"`。`openai_api_key`参数设置为你的 OpenAI API 密钥，这是验证和访问 OpenAI API 所必需的。`temperature`参数设置为零，这意味着模型将生成确定性的响应：

```
# 创建一个 ChatOpenAI 对象
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", openai_api_key="Your OpenAI Key", temperature=0)
```

### 第 4 步：定义生成响应的函数

你必须定义一个名为`generate_response`的函数，它接受一个参数`text`，用于根据输入文本从聊天模型生成响应：

```
# 创建一个生成响应的函数
def generate_response(text):
    messages = [HumanMessage(content=text)]
    response = chat_model.invoke(messages)
    return response.content
```

让我们进一步讨论上面的代码：

-   **`messages = [HumanMessage(content=text)]`**：在`generate_response`函数内部，你创建了一个名为`messages`的列表，其中包含一个`HumanMessage`对象。`HumanMessage`类代表由人类用户发送的消息，它接受输入文本作为其内容。
-   **`response = chat_model.invoke(messages)`**：你使用`ChatOpenAI`对象的`invoke`方法调用聊天模型。你将包含人类消息的`messages`列表作为参数传递。聊天模型处理该消息并生成响应。
-   **`return response.content`**：你返回生成的响应的内容。聊天模型返回的响应对象有一个`content`属性，其中包含响应的实际文本。

### 第 5 步：创建主交互循环

下面，你启动一个无限循环，允许用户重复与聊天机器人交互：

```
# 创建一个与聊天机器人交互的循环
while True:
    # 获取用户输入
    user_input = input("Enter a message: ")
    # 生成响应
    response = generate_response(user_input)
```



#### 打印响应

`print(response)`

以下是聊天机器人的完整端到端工作代码：

```
!pip install openai==1.35.13 langchain==0.2.7 langchain_openai==0.1.16
!pip show openai langchain langchain_openai
```

