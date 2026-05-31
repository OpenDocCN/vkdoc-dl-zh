# 结构化聊天代理

让我向你介绍结构化聊天代理，这是一个使用多输入工具的强大工具。首先，确保你已经安装了必要的库。在这个例子中，你将使用 LangChain、Tavily Search 和 OpenAI 的 ChatGPT。以下是如何从 LangChain 库和 LangChain 社区工具中导入它们：

```python
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
```

### 初始化工具

在这个例子中，你将使用 Tavily Search 工具来测试代理，以便在线搜索信息。这行代码创建了一个代理可以访问的工具列表。你将 `max_results` 参数设置为 `1`，表示该工具最多返回一个搜索结果：

```python
tools = [TavilySearchResults(max_results=1)]
```

## 创建代理

现在，通过使用 `hub.pull()` 从 LangChain 中心拉取提示词来创建你的结构化聊天代理，如下面第一行所示。然后，你通过为业务分析师助手提供具体的指令和示例来自定义这个提示词，以满足你的需求：

```python
prompt = hub.pull("hwchase17/structured-chat-agent")
prompt.messages[0].prompt.template = """
你是一名业务分析师助手，任务是为企业家和企业主提供明智的决策支持。
使用提供的搜索工具查找相关信息，并尽最大努力回答用户的问题。
如果问题无法通过搜索结果回答，请指导用户在哪里可以找到更多信息。
请以清晰、简洁且结构良好的格式提供你的回复。

以下是你可能会被问到的一些业务相关问题的示例：
- 小企业有哪些有效的营销策略？
- 我如何改善公司的现金流？
- 制定商业计划的关键步骤是什么？
- 如何为新产品创意进行市场调研？
- 初创企业面临哪些常见挑战，以及如何克服它们？

请记住提供针对用户特定需求和情况的可操作建议。
让我们一起努力帮助企业蓬勃发展！
"""
```

接下来，你将选择驱动我们代理的语言模型（LLM）。在这个例子中，你将使用 OpenAI 的 ChatGPT，温度设置为 `0`，模型为 `gpt-3.5-turbo-1106`：

```python
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")
```

最后，你将通过调用 `create_structured_chat_agent` 函数来构建代理，传入 LLM、工具和提示词：

```python
agent = create_structured_chat_agent(llm, tools, prompt)
```

### 定义辅助函数

然后，你创建一个辅助函数来处理语言模型生成的输出。它检查输出是否包含字符串 `"Invalid or incomplete response"`，如果包含则引发 `ValueError` 异常。否则，它按原样返回输出：

```python
def process_llm_output(output):
    if "Invalid or incomplete response" in output:
        raise ValueError("The language model generated an invalid or incomplete response.")
    return output
```

## 运行代理

代理准备就绪后，创建一个 `AgentExecutor` 类的实例来运行它。你提供代理、工具以及各种配置选项，例如 `verbose`（用于详细输出）、`handle_parsing_errors`（用于优雅地处理解析错误）、`max_iterations`（用于限制最大迭代次数）和 `early_stopping_method`（用于指定提前停止的方法）：

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method="force",
)
```

现在，向你的代理提问，看看它的表现：

```python
question = "What are some effective ways to reduce operational costs in a manufacturing business?"
```

最后，你使用 try-except 块来执行代理并处理任何潜在错误。你使用 `agent_executor` 来调用代理。



`invoke({"input": question})` 并将问题作为输入传入。代理会生成一个结果，你需要使用 `process_llm_output()` 函数来处理该结果。如果输出有效，则打印处理后的结果。如果由于响应无效或不完整而引发 `ValueError`，则捕获该异常并打印一条错误消息。

观察代理如何使用 Tavily Search 搜索关于 LangChain 的信息，并提供一个简洁的摘要。现在你可以忽略任何错误，因为这超出了练习的范围。你现在已经学会了如何使用 LangChain 设置和运行一个结构化聊天代理来回答业务相关问题。

```
**> 进入新的 AgentExecutor 链...**

*{*

*"response": "降低制造业企业的运营成本可以通过多种策略实现：",*

*"strategies": [*

*{*

*"title": "精益生产",*

*"description": "实施精益生产原则以消除浪费、提高效率并降低成本。这包括简化流程、优化库存水平和最小化停机时间。"*

*},*

*{*

*"title": "能源效率",*

第 9 章 构建不同类型的代理

*"description": "投资于节能设备和流程以降低公用事业费用。进行能源审计以确定改进领域，并考虑可再生能源。"*

*},*

*{*

*"title": "供应商谈判",*

*"description": "与供应商谈判以获得更优惠的价格、折扣或有利的付款条件。与主要供应商整合采购以利用批量折扣。"*

*},*

*{*

*"title": "库存管理",*

*"description": "优化库存水平以降低持有成本，并最小化库存过剩或过时的风险。在可行的情况下实施准时制库存实践。"*

*},*

*{*

*"title": "非核心业务外包",*

*"description": "考虑将非核心职能（如清洁服务、维护或某些制造流程）外包给专业第三方提供商，以降低间接成本。"*

*},*

*{*

*"title": "流程自动化",*

*"description": "投资自动化技术以简化生产流程、提高准确性并降低劳动力成本。这可能涉及机器人技术、自动化装配线或软件系统。"*

第 9 章 构建不同类型的代理

*}*

*]*

*}*
```

### 使用聊天历史

你可以使用与前两种代理类型相同的方法来利用聊天历史，以获得更具对话性的体验。通过传入之前的对话轮次，你的代理可以以更具上下文关联性和更自然的方式进行响应。

## ReAct 代理

接下来是 ReAct 代理，这是一个强大的工具，允许你在 AI 应用中实现 ReAct 逻辑。它使代理能够根据收集到的信息进行推理和行动。

首先，确保你已经安装了必要的库。在本例中，你将使用 LangChain、Tavily Search 和 OpenAI。以下是导入它们的方法：

```python
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI
```

### 初始化工具

首先，让我们为 ReAct 代理加载一些工具以供使用。在此示例中，你将使用 Tavily Search 来允许你的代理在线搜索信息：

```python
tools = [TavilySearchResults(max_results=1)]
```

### 创建代理

现在，通过从 LangChain Hub 拉取提示来创建你的 ReAct 代理。你可以根据需要自定义此提示，但这里我们将使用默认提示：

```python
prompt = hub.pull("hwchase17/react")
```

接下来，你将选择要使用的语言模型 (LLM)。在本例中，我们使用 OpenAI：

```python
llm = OpenAI()
```

最后，通过调用 `create_react_agent` 函数，传入 LLM、工具和提示来构建 ReAct 代理：

```python
agent = create_react_agent(llm, tools, prompt)
```

### 运行代理

准备好你的 ReAct 代理后，创建一个执行器并通过传入代理、工具并设置 `verbose=True` 来运行它，以便查看代理的思考过程：

```python
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

现在，向你的代理提问并观察它的运行：

```python
agent_executor.invoke({"input": "what is LangChain?"})
```

观察代理如何经历一系列思考和行动来收集关于 LangChain 的信息：

```
我应该研究 LangChain 以了解更多信息。
Action: tavily_search_results_json
Action Input: "LangChain"

我应该阅读摘要并查看 LangChain 的不同功能和集成。
Action: tavily_search_results_json
Action Input: "LangChain features and integrations"

我应该记下 LangChain 的发布日期和受欢迎程度。
Action: tavily_search_results_json
Action Input: "LangChain launch date and popularity"

我现在知道最终答案了。
Final Answer: LangChain 是一个开源编排框架，用于构建使用大型语言模型 (LLM) 的应用程序，例如聊天机器人和虚拟代理。它由 Harrison Chase 于 2022 年 10 月推出，并于 2023 年 6 月成为 GitHub 上增长最快的开源项目。
```

### 使用聊天历史

当将 ReAct 代理与聊天历史一起使用时，你需要一个考虑到这一点的提示。让我们从 LangChain Hub 拉取聊天专用的提示：

```python
prompt = hub.pull("hwchase17/react-chat")
```

现在，让我们使用这个提示来构建 ReAct 代理：

```python
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

要使用聊天历史，你可以传入一个表示先前对话轮次的字符串。示例如下：

```python
from langchain_core.messages import AIMessage, HumanMessage

agent_executor.invoke(
    {
        "input": "what's my name? Only use a tool if needed, otherwise respond with Final Answer",
    }
)
```



