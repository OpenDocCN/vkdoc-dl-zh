# OpenAI 工具代理

OpenAI 有两个相关概念：“函数”和“工具”。`functions` 允许你的代理调用单个函数，而 `tools` 则使其能够在适当时调用一个或多个函数。在 OpenAI 聊天 API 中，`functions` 现在被视为一个遗留选项，已被弃用，取而代之的是 `tools`。

因此，如果你正在使用 OpenAI 模型创建代理，你应该使用 OpenAI 工具代理，而不是 OpenAI 函数代理。

使用 `tools` 有一个显著的优势，因为它允许模型在适当时请求调用多个函数。这有助于减少代理达成目标所需的时间，使其更加高效和有效。

现在，让我们看看代码，了解如何实际操作创建一个 OpenAI 工具代理。

首先，确保你已经安装了必要的库：

```
%pip install --upgrade --quiet langchain-openai tavily-python
```

接下来，你必须导入所需的模块：

```python
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
```

### 初始化工具

在这个例子中，你将赋予我们的代理使用 Tavily 搜索网络的能力：

```python
tools = [TavilySearchResults(max_results=1)]
```

## 创建代理

现在，你将通过从 LangChain 中心拉取提示词并选择驱动代理的 LLM 来创建你的 OpenAI 工具代理：

```python
prompt = hub.pull("hwchase17/openai-tools-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
agent = create_openai_tools_agent(llm, tools, prompt)
```

## 运行代理

代理准备就绪后，创建一个执行器来运行它，并用一个查询来调用它：

```python
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "what is LangChain?"})
```

观察你的代理如何开始行动，使用 Tavily 搜索工具查找关于 LangChain 的信息并提供简洁的摘要：

> LangChain 是一个开源编排框架，用于开发使用大型语言模型的应用程序。它本质上是一个为 Python 和 Javascript 提供的抽象库，代表了常见的步骤和概念。LangChain 简化了编程以及与外部数据源和软件工作流集成的过程。它支持各种大型语言模型提供商，包括 OpenAI、Google 和 IBM。你可以在 IBM 网站上找到关于 LangChain 的更多信息：[LangChain - IBM](https://www.ibm.com/topics/langchain)

### 使用聊天历史

OpenAI 工具代理最酷的功能之一是它们能够利用聊天历史来实现更具对话性的体验。通过传入之前的对话轮次，你的代理可以以更具上下文和更自然的方式做出响应。

以下是如何在你的代理中使用聊天历史的示例：

```python
from langchain_core.messages import AIMessage, HumanMessage

agent_executor.invoke(
    {
        "input": "what's my name? Don't use tools to look this up unless you NEED to",
        "chat_history": [
            HumanMessage(content="hi! my name is Rabi Jay"),
            AIMessage(content="Hello Rabi Jay! How can I assist you today?"),
        ],
    }
)
```

在这种情况下，代理会记住之前的对话并做出相应回应：

> Your name is Rabi Jay.

你刚刚见证了 OpenAI 工具代理的强大功能。它们可以智能地选择和使用函数，提供结构化输出，甚至可以使用聊天历史进行对话式交互。

