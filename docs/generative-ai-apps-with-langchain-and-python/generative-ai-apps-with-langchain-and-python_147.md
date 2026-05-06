# 第 10 章：项目：为常见用例构建代理应用

在本章中，我们将探讨如何使用 LangChain 创建自定义代理。到本章结束时，您将对如何加载语言模型、定义工具、创建提示以及将所有内容绑定在一起以构建一个功能性的代理有扎实的理解。我们还将涵盖实际用例，如客户支持自动化、个性化推荐以及实时数据分析和决策。

## 创建自定义代理

现在，让我们开始创建一个自定义代理。

© Rabi Jay 2024
R. Jay, *Generative AI Apps with LangChain and Python*,
[`doi.org/10.1007/979-8-8688-0882-1_10`](https://doi.org/10.1007/979-8-8688-0882-1_10)

### 加载语言模型

第一步是加载语言模型。在本例中，您将使用 OpenAI 的 `ChatOpenAI` 模型，但欢迎您在后续过程中尝试其他模型：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
```

### 定义工具

接下来，您需要为代理配备工具。让我们从一个简单的 Python 函数开始，该函数计算给定单词的长度：

```python
from langchain.agents import tool

@tool
def get_word_length(word: str) -> int:
    """返回一个单词的长度。"""
    return len(word)

get_word_length.invoke("abc")  # 输出：3
```

请密切注意这里的文档字符串。它作为关键指南，帮助您的代理理解如何有效地使用该工具。

现在，创建一个列表来保存您的代理将拥有的所有工具：

```python
tools = [get_word_length]
```

### 创建提示

有了语言模型和工具，现在是时候精心设计将指导代理行为的提示了。得益于 OpenAI 的函数调用，您可以保持简单，专注于代理所需的基本信息：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个非常强大的助手，但不知道当前事件",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
```

您为代理提供了一个简短的系统消息、一个用户输入的占位符以及一个代理草稿板的占位符（用于存储中间步骤和工具输出的空间）。

### 将工具绑定到语言模型

然后，您必须将工具绑定到语言模型，这有助于您的代理了解其具备哪些能力：

```python
llm_with_tools = llm.bind_tools(tools)
```

### 创建代理

导入几个实用函数，以帮助格式化代理的中间步骤并解析其输出：

```python
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
```

就这样，您的代理准备好了！

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

您已经创建了一个 `AgentExecutor` 来与您的代理交互并观察其思考过程。

### 测试您的代理

现在，让我们来测试您的代理！您将向它提出一个简单的问题，并观察它的表现：

```python
list(agent_executor.stream({"input": "单词 first 中有多少个字母"}))
```

您应该会看到类似以下的输出：

```
> 进入新的 AgentExecutor 链...
调用：`get_word_length`，参数为 `{'word': 'eudca'}`
单词 "first" 中有 5 个字母。
> 链结束。
```

太棒了！您的代理成功使用了 `get_word_length` 工具来回答您的问题。

### 添加记忆

但是，如果您希望您的代理记住之前的交互并进行更自然的对话呢？

首先，您需要在提示中添加一个聊天历史的占位符：

```python
MEMORY_KEY = "chat_history"

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个非常强大的助手，但不擅长计算单词的长度。",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
```

接下来，设置一个列表来跟踪聊天历史：

```python
from langchain_core.messages import AIMessage, HumanMessage

chat_history = []
```

最后，更新您的代理和 `AgentExecutor` 以包含聊天历史：

```python
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

现在，在与您的代理交互时，将输入和输出作为聊天历史进行跟踪：

```python
input1 = "单词 first 中有多少个字母？"
result = agent_executor.invoke({"input": input1, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=input1),
        AIMessage(content=result["output"]),
    ]
)
agent_executor.invoke({"input": "那是一个真实的单词吗？", "chat_history": chat_history})
```

您的代理现在可以进行来回对话，记住之前的交互并提供上下文相关的响应。

您刚刚使用 LangChain 创建了自己的自定义代理。

## 代理的实际用例

让我们回顾一下代理的一些实际用例。

### 客户支持自动化

在本节中，让我们探讨如何使用代理来自动化和简化您的客户支持流程。如果您有一个客户群不断增长的企业，并且希望在不让团队不堪重负的情况下提供一流的支持，那么这个用例可以帮到您。您可以创建一个自助服务支持系统，高效且有效地处理常见的客户咨询。

让我们回顾一下流程中的步骤：

1. **识别常见的客户咨询：**
   - 分析您的客户支持数据，以识别最常见的问题和常见问题。
   - 将这些咨询分类到不同的主题或类别中，例如产品信息、故障排除、计费等。

2. **创建知识库：**
   - 编写一个全面的知识库，涵盖已识别的主题，并为常见问题提供清晰、简洁的答案。
   - 将知识库组织成结构化格式，例如常见问题解答页面、产品文档或故障排除指南。

3. **设置代理：**
   - 使用 LangChain 代理框架构建您的客户支持自动化系统。
   - 安装并导入必要的库：

```python
%pip install --upgrade --quiet langchain-openai tavily-python langchain_community langchain_openai

from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
```

4. **定义工具和记忆：**
   - 创建允许代理访问和检索知识库信息的工具：

```python
def search_knowledge_base(query):
```



