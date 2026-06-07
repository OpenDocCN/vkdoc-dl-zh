# 第 9 章 构建不同类型的代理

## 工具调用代理

让我向你介绍一个名为工具调用的强大概念，它能帮助你的代理检测何时应调用一个或多个工具，并返回这些工具所需的适当输入。

借助工具调用，代理可以智能地选择正确的工具，并确切知道每个工具需要哪些参数。在 API 世界中，你可以向代理描述这些工具，它会输出一个包含调用工具所需参数的结构化对象（如 JSON）。

工具调用的目标是确保你的代理可靠地返回有效且有用的工具调用，超越通用文本补全或聊天 API 的能力。通过利用这种结构化输出并允许代理从多个工具中选择，你可以创建一个反复调用工具并接收结果，直到解决查询的代理。

### 设置

要开始使用工具调用，你需要一个支持该功能的模型。

`LangChain` 提供了广泛的选择，包括 `Anthropic`、`Google Gemini`、`Mistral` 和 `OpenAI`。你可以在 `LangChain` 文档中查看支持的模型。

在本演示中，你将使用 `Tavily`，但也可以随意替换为任何其他内置工具，甚至添加你自己的自定义工具。要使用 `Tavily`，你需要注册一个 API 密钥，并将其设置为 `process.env.TAVILY_API_KEY`。

首先，让我们安装必要的依赖项：

```bash
pip install -qU langchain-openai langchain tavily-python
```

接下来，将你的 OpenAI API 密钥设置为环境变量：

```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()
os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter your Tavily API key: ")
```

现在，你将导入 `langchain_openai` 中的 `ChatOpenAI` 类，并创建一个语言模型实例：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
```

### 初始化工具

让我们创建一个可以使用 `Tavily` 搜索网络的工具：

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

tools = [TavilySearchResults(max_results=1)]
```

## 创建代理

现在，你需要通过定义提示并创建代理来初始化你的工具调用代理：

```python
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Make sure to use the tavily_search_results_json tool for information.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
```

## 运行代理

代理初始化后，你将创建一个执行器来运行它，并使用查询调用它：

```python
# 通过传入代理和工具创建代理执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 使用与业务相关的问题调用代理执行器
response = agent_executor.invoke({"input": "How can our company reduce operational costs by leveraging AI technologies?"})

#### 打印响应
print(response)
```

观察你的代理如何立即行动，使用 `Tavily` 搜索工具查找与问题相关的信息，并提供简洁的摘要：

```
> Entering new AgentExecutor chain...
***Could not parse LLM output: {***
***"response": "Reducing operational costs by leveraging AI technologies can be achieved through various strategies:",***
***"strategies": [***
***{***
***"title": "Automating Routine Tasks",***
***"description": "Implement AI-powered automation to handle repetitive and time-consuming tasks such as data entry, customer support inquiries, and inventory management. This can reduce the need for manual labor and free up employees to focus on higher-value activities."***
***},***
***{***
***"title": "Predictive Maintenance",***
***"description": "Utilize AI for predictive maintenance of equipment and machinery. By analyzing data from sensors and historical maintenance records, AI can predict potential failures, allowing for proactive maintenance and minimizing downtime."***
***},***
***{***
***"title": "Optimizing Supply Chain Management",***
***"description": "AI can optimize supply chain operations by analyzing demand patterns, predicting inventory needs, and identifying the most cost-effective shipping routes. This can lead to reduced inventory holding costs and improved efficiency."***
***},***
***{***
***"title": "Enhancing Energy Efficiency",***
***"description": "Implement AI-powered systems to optimize energy usage within facilities. AI can analyze patterns of energy consumption and automatically adjust settings for lighting, heating, and cooling, leading to cost savings."***
***},***
***{***
***"title": "Improving Decision Making",***
***"description": "AI can provide valuable insights by analyzing large datasets and identifying cost-saving opportunities. By leveraging AI for data-driven decision making, companies can optimize processes and resource allocation."***
***}***
***]***
***}***
```

### 使用聊天历史

工具调用代理最酷的功能之一是能够利用聊天历史实现更对话式的体验。通过传入之前的对话轮次，你的代理可以以更具上下文和更自然的方式做出响应。

以下是如何将聊天历史与代理结合使用的示例：

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

在这种情况下，代理会记住之前的对话并相应做出响应：

```
Based on what you told me, your name is Rabi Jay. I don't need to use any tools to look that up since you directly provided your name.
```

你刚刚见证了工具调用代理的强大功能。它们可以智能地选择和使用工具，提供结构化输出，甚至利用聊天历史进行对话式交互。

## OpenAI 工具

让我们来谈谈 OpenAI 的一项激动人心的功能，称为"工具"，它帮助你的代理检测何时应调用一个或多个函数，并返回适当的输入。较新的 OpenAI 模型已针对此能力进行了微调，使你的代理更智能、更高效。

在 API 调用中，你可以向代理描述函数，它会智能地选择输出一个包含调用这些函数所需参数的 JSON 对象。

OpenAI 工具的目标是确保你的代理可靠地返回有效且有用的函数调用，超越通用文本补全或聊天 API 的能力。它使你的代理更加精确和高效。