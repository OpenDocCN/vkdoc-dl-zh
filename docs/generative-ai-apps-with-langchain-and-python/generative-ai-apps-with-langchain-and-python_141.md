# 请注意 `chat_history` 是一个字符串，因为此提示词面向的是大语言模型，而非聊天模型

`"chat_history": "Human: Hi! My name is Rabi\nAI: Hello Rabi! Nice to meet you"`

在这种情况下，智能体将根据给定的聊天历史来推理是否需要使用工具：

`Thought: Do I need to use a tool? No`

`Final Answer: Your name is Rabi.`

正在进入新的 `AgentExecutor` 链…

**注意** 根据您使用的模型不同，您也可能会收到不同的消息。以下是一个示例：

`as an ai, i don’t have access to personal data about individuals unless it has been shared with me in the course of our conversation. i am designed to respect user privacy and confidentiality. therefore, i don’t know the user’s name. final answer: i’m sorry, but i don’t have access to that information.`

> 链已结束。

## 第 9 章 构建不同类型的智能体

`{'input': "what's my name? only use a tool if needed, otherwise respond with final answer", 'chat_history': 'human: hi! My name is rabi\nai: hello rabi! nice to meet you', 'output': "i'm sorry, but i don't have access to that information."}`

## 自问自答智能体

让我们来看看具备搜索能力的自问自答智能体，它可以帮助您找到迫切问题的答案。

首先，请确保您的工具箱中拥有必要的工具。

在本例中，您将使用 `LangChain`、`Fireworks LLM` 和 `Tavily Answer`。请继续导入它们：

```python
from langchain import hub
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent
from langchain_community.llms import Fireworks
from langchain_community.tools.tavily_search import TavilyAnswer
```

### 初始化工具

现在，初始化您的自问自答智能体将要使用的工具。对于此智能体，您将使用 `Tavily Answer`，它可以直接回答您的问题。

需要注意的一点是，此智能体只能使用一个工具，并且该工具必须命名为 `"Intermediate Answer"`。那么，让我们来设置它：

```python
tools = [TavilyAnswer(max_results=1, name="Intermediate Answer")]
```

### 创建智能体

工具准备就绪后，就可以创建您的自问自答智能体了。您将从 `LangChain` 中心拉取提示词开始。您可以根据需要自定义此提示词，但这里我们将使用默认提示词：

```python
prompt = hub.pull("hwchase17/self-ask-with-search")
```

接下来，选择驱动智能体思考过程的大语言模型。在本示例中，您将使用 `Fireworks LLM`：

设置 `Fireworks` API 密钥：

```python
os.environ["FIREWORKS_API_KEY"] = "your-fireworks-api-key-here"
```



