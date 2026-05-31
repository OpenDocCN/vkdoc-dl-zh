# 初始化 Fireworks 大语言模型

```python
llm = Fireworks(
    model="accounts/fireworks/models/llama-v2-13b-chat",  # 选择合适的模型
    max_tokens=1024,
    temperature=0.7
)
```

最后，通过调用 `create_self_ask_with_search_agent` 函数，传入大语言模型、工具和提示词，构建带有搜索功能的自我提问代理：

```python
agent = create_self_ask_with_search_agent(llm, tools, prompt)
```

## 运行代理

组装好自我提问代理后，创建一个执行器并传入代理、工具，同时设置 `verbose=True` 来查看代理的思考过程，使其真正运行起来：

```python
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

现在，向代理提问，观察它如何施展魔法：

```python
agent_executor.invoke(
    {"input": "科技行业市值最高的公司总部在哪里？"}
)
```

代理会先向自己提出一个后续问题，以收集更多信息：

是的。

后续问题：科技行业中哪家公司市值最高？

使用 Tavily 回答工具，代理会找到自己问题的答案：

截至 2024 年 6 月 18 日，市值最高的公司是 NVIDIA。

掌握了这些信息后，代理会给出最终答案：

所以最终答案是：加利福尼亚州圣克拉拉。

带有搜索功能的自我提问代理通过将问题分解为更小的步骤并使用可用工具，成功找到了问题的答案。

## 自主决策能力

让我们看一个更高级的例子，展示代理的自主决策能力。在这个例子中，你将创建一个“任务管理器”代理，它能够理解用户请求，将其分解为子任务，并自主决定使用哪些工具来完成每个子任务。

以下是逐步的代码分解：

1. 安装必要的库：

```bash
pip install langchain==0.2.5 openai==1.35.13 google-search-results serpapi
```

2. 导入所需的模块：

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper
```

3. 设置语言模型和搜索工具：

```python
llm = OpenAI(temperature=0)
search = GoogleSearchAPIWrapper()
```

你正在使用 OpenAI 语言模型和 Google 搜索 API，使代理能够在线搜索信息。

4. 定义代理可用的工具：

```python
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="用于在互联网上搜索信息。"
    )
]
```

在这个例子中，你定义了一个名为“Search”的工具，允许代理使用 Google 搜索 API 执行互联网搜索。

5. 初始化代理：

```python
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)
```

你使用定义好的工具、语言模型和“zero-shot-react-description”代理类型来初始化代理，这使代理能够根据用户的请求自主决定使用哪些工具。

6. 向代理提供一个任务：

```python
task = "我需要计划一次巴黎之旅。我应该参观哪些顶级旅游景点，以及一年中什么时候去最好？"
result = agent.run(task)
print(result)
```

你给代理一个与计划巴黎之旅相关的任务。代理将自主分解任务，搜索相关信息，并提供全面的回答。

当你运行这段代码时，代理将自主处理任务并提供类似以下的回答：

为了计划你的巴黎之旅，以下是应该参观的顶级旅游景点以及一年中最佳出行时间：

巴黎的顶级旅游景点：

1. 埃菲尔铁塔 - 标志性地标，可欣赏城市壮丽景色
2. 卢浮宫 - 世界著名的艺术博物馆，收藏了《蒙娜丽莎》和其他杰作
3. 巴黎圣母院 - 以哥特式建筑闻名的历史大教堂（因 2019 年火灾目前正在修复中）
4. 凯旋门 - 纪念为法国战斗和牺牲者的著名纪念碑
5. 凡尔赛宫 - 华丽的昔日皇家住所，拥有令人惊叹的花园
6. 蒙马特 - 迷人的山顶街区，以其艺术历史和圣心大教堂而闻名
7. 塞纳河游船 - 沿着塞纳河乘船游览，欣赏城市地标

游览巴黎的最佳时间：

游览巴黎的最佳时间取决于你的偏好和优先事项。以下是按季节划分的说明：

- 春季（3 月至 5 月）：天气温和，鲜花盛开，与夏季相比游客较少。非常适合户外活动和观光。
- 夏季（6 月至 8 月）：天气温暖至炎热，白天时间长，是旅游旺季。非常适合户外活动和节日，但预计会有更多游客和更高的价格。
- 秋季（9 月至 11 月）：天气宜人，游客较少，秋叶美丽。非常适合观光、文化活动和美食节。
- 冬季（12 月至 2 月）：天气寒冷，白天较短，节日装饰喜庆。适合室内活动、博物馆和圣诞市场。预计价格较低，游客较少。

总的来说，春季和秋季的平季提供了宜人的天气、可控的游客数量和合理价格的良好平衡。然而，巴黎是一个全年皆宜的目的地，每个季节都有其独特的魅力。

## 使用多种工具执行任务的智能代理

在本节中，我们将创建一个能够使用多种工具执行任务的智能代理。该代理将在线搜索信息，并从预加载的索引中查找特定数据。

### 设置：介绍 LangSmith

构建代理可能很棘手，尤其是在调试和可观测性方面。这就是 LangSmith 发挥作用的地方，它通过追踪代理工作流程中的所有步骤，使构建和调试代理的过程变得简单。要设置 LangSmith，你只需要设置几个环境变量：

```bash
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="<your-api-key>"
```

确保将 `<your-api-key>` 替换为你实际的 LangSmith API 密钥。

### 定义工具

我们将为你的代理配备两个强大的工具——用于在线搜索的 Tavily 和用于查询本地索引的检索器。

#### 工具 1：Tavily

Tavily 是 LangChain 中的一个内置工具，允许你的代理轻松搜索网络，并访问庞大的知识库。要使用 Tavily，你需要一个 API 密钥。他们提供免费套餐，但如果你没有或不想创建，可以跳过此步骤。

获得 Tavily API 密钥后，将其导出为环境变量：

```bash
export TAVILY_API_KEY="..."
```

现在，让我们创建一个 `TavilySearchResults` 工具的实例：

```python
from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults()
```

你可以轻松地使用查询来调用搜索工具：

```python
search.invoke("旧金山的天气怎么样")
```

这将返回与旧金山天气相关的搜索结果列表。

#### 工具 2：检索器

除了在线搜索，创建一个检索器，允许你的代理从本地索引中查找信息。当你希望代理快速访问特定数据时，这尤其有用。

要创建检索器，请按照以下步骤操作：

1. 使用 `WebBaseLoader` 加载数据：

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
```

2. 使用 `RecursiveCharacterTextSplitter` 将加载的文档分割成块：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
```

3. 使用 FAISS 和 `OpenAIEmbeddings` 创建向量存储：

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()
```

现在你有了一个可以在索引文档中搜索信息的检索器。你可以使用查询来调用检索器：

```python
retriever.invoke("如何上传数据集")[0]
```

这将根据查询返回最相关的文档块。

#### 创建检索器工具

为了使你的代理更容易使用检索器，我们可以使用 LangChain 中的 `create_retriever_tool` 函数将其转换为工具：

```python
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "搜索关于 LangSmith 的信息。对于任何关于 LangSmith 的问题，你必须使用此工具！",
)
```

这将创建一个具有特定名称和描述的检索器工具，使你的代理更容易理解何时以及如何使用它。

### 整合所有内容

现在你已经准备好了工具，创建一个包含 Tavily 搜索工具和检索器工具的列表：

```python
tools = [search, retriever_tool]
```

这允许你的代理执行在线搜索并从本地索引中查找信息。

### 选择大语言模型

下一步是选择将作为代理大脑的语言模型。在这个例子中，你将使用 OpenAI Functions 代理，这是一个强大且多功能的选项。

首先，从 `langchain_openai` 模块导入 `ChatOpenAI` 类并创建大语言模型实例：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
```

这里，你使用的是 `gpt-3.5-turbo-0125` 模型，温度为 0。你可以根据具体需求调整这些参数。

### 选择提示词

接下来，你需要选择将指导代理行为的提示词。提示词在塑造代理如何与工具交互以及处理输入方面起着至关重要的作用。

如果你可以访问 LangSmith，可以通过访问 [`smith.langchain.com/hub/hwchase17/openai-functions-agent`](https://smith.langchain.com/hub/hwchase17/openai-functions-agent) 来探索预定义提示词的内容。

或者，你可以使用 LangChain 的 `hub` 模块来拉取提示词：

```python
from langchain import hub

prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages
```

这将检索提示词模板并显示其消息，包括系统消息、聊天历史和代理草稿板的占位符以及用户输入。

### 初始化代理

现在，是时候通过结合我们之前定义的大语言模型、提示词和工具来初始化你的代理了。代理负责接收输入并根据该输入决定采取什么行动。需要注意的是，代理本身不执行行动——这是 `AgentExecutor` 的工作，我们将在下一步讨论。

要创建代理，你将使用 `langchain.agents` 模块中的 `create_tool_calling_agent` 函数：

```python
from langchain.agents import create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt)
```

此函数接收大语言模型、工具列表和提示词，并返回一个已初始化的代理，随时可以处理我们的任务。

### 创建 AgentExecutor

最后一块拼图是 `AgentExecutor`，它将代理（大脑）和工具（能力）结合在一起。`AgentExecutor` 重复调用代理以确定下一步行动，然后执行相应的工具。

要创建 `AgentExecutor`，你将使用以下代码：

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

我们传入初始化的代理、工具列表，并设置 `verbose=True` 以在执行期间启用详细输出。

让我们测试一下代理，看看它能做什么！你将运行几个查询，并观察代理如何处理它们。请记住，目前这些查询是无状态的，意味着代理不会记住之前的交互。

首先，从一个简单的问候开始：

```python
agent_executor.invoke({"input": "你好！"})
```

当你运行这段代码时，代理将处理输入并生成响应。以下是我得到的结果：

```
[1m> 进入新的 AgentExecutor 链...[0m
[32;1m[1;3m 你好！今天我能帮你什么？[0m
[1m> 链结束。[0m
{'input': '你好！', 'output': '你好！今天我能帮你什么？'}
```

如你所见，代理以友好的问候回应，询问如何帮助我们。`verbose=True` 设置允许我们看到代理的思考过程，以绿色文本表示。

现在，让我们问代理一个与 LangSmith 和测试相关的更具体的问题：

```python
agent_executor.invoke({"input": "langsmith 如何帮助进行测试？"})
```

以下是响应：

```
[1m> 进入新的 AgentExecutor 链...[0m
[32;1m[1;3m
调用：`langsmith_search`，参数为 `{'query': 'LangSmith 如何帮助进行测试'}`
[0m[33;1m[1;3mLangSmith 是一个用于构建生产级大语言模型应用的平台，可以通过以下方式帮助进行测试：

1. **追踪**：LangSmith 提供追踪功能，允许你在测试期间密切监控和评估你的应用。你可以记录追踪信息以跟踪应用的行为并识别任何问题。
2. **评估**：LangSmith 提供评估功能，使你能够在测试期间评估应用的性能。这有助于确保你的应用按预期运行并满足所需标准。
3. **生产监控与自动化**：LangSmith 允许你在生产中监控你的应用并自动化某些流程，这对于测试不同场景和确保应用的稳定性非常有益。
4. **提示词中心**：LangSmith 包含一个提示词中心，这是一个提示词管理工具，通过为管理应用的提示词和输入提供集中位置，可以简化测试过程。

总的来说，LangSmith 可以通过提供监控、评估和自动化流程的工具来协助测试，以确保你的应用在测试阶段的可靠性和性能。[0m
[1m> 链结束。[0m
{'input': 'langsmith 如何帮助进行测试？', 'output': 'LangSmith 是一个用于构建生产级大语言模型应用的平台，可以通过以下方式帮助进行测试：\n\n1. **追踪**：LangSmith 提供追踪功能，允许你在测试期间密切监控和评估你的应用。你可以记录追踪信息以跟踪应用的行为并识别任何问题。\n\n2. **评估**：LangSmith 提供评估功能，使你能够在测试期间评估应用的性能。这有助于确保你的应用按预期运行并满足所需标准。\n\n3. **生产监控与自动化**：LangSmith 允许你在生产中监控你的应用并自动化某些流程，这对于测试不同场景和确保应用的稳定性非常有益。\n\n4. **提示词中心**：LangSmith 包含一个提示词中心，这是一个提示词管理工具，通过为管理应用的提示词和输入提供集中位置，可以简化测试过程。\n\n 总的来说，LangSmith 可以通过提供监控、评估和自动化流程的工具来协助测试，以确保你的应用在测试阶段的可靠性和性能。'}
```

在这个例子中，代理调用了 `langsmith_search` 工具，查询参数为“LangSmith 如何帮助进行测试”。它从索引文档中检索相关信息，并生成详细的响应，解释 LangSmith 的功能（如追踪、评估、生产监控和提示词中心）如何帮助测试大语言模型应用。

最后，让我们问代理旧金山的天气：

```python
agent_executor.invoke({"input": "旧金山的天气怎么样？"})
```

以下是响应：

```
[1m> 进入新的 AgentExecutor 链...[0m
[32;1m[1;3m
调用：`tavily_search_results_json`，参数为 `{'query': 'San Francisco 天气'}`
[0m[36;1m[1;3m[{'url': 'https://www.weatherapi.com/', 'content': "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.78, 'lon': -122.42, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1712847697, 'localtime': '2024-04-11 8:01'}, 'current': {'last_updated_epoch': 1712847600, 'last_updated': '2024-04-11 08:00', 'temp_c': 11.1, 'temp_f': 52.0, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 2.2, 'wind_kph': 3.6, 'wind_degree': 10, 'wind_dir': 'N', 'pressure_mb': 1015.0, 'pressure_in': 29.98, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 97, 'cloud': 25, 'feelslike_c': 11.5, 'feelslike_f': 52.6, 'vis_km': 14.0, 'vis_miles': 8.0, 'uv': 4.0, 'gust_mph': 2.8, 'gust_kph': 4.4}}"}][0m[32;1m[1;3m 旧金山目前的天气是局部多云，温度为 52.0°F（11.1°C）。风速为 3.6 公里/小时，来自北方，湿度为 97%。[0m
[1m> 链结束。[0m
{'input': '旧金山的天气怎么样？', 'output': '旧金山目前的天气是局部多云，温度为 52.0°F（11.1°C）。风速为 3.6 公里/小时，来自北方，湿度为 97%。'}
```

对于这个查询，代理调用了 `tavily_search_results_json` 工具来搜索旧金山的天气信息。它从搜索结果中检索相关数据，并呈现当前天气状况的简明摘要，包括温度、风速和湿度。

这些例子展示了我们的代理如何处理不同类型的查询，并使用适当的工具生成信息丰富的响应。

在这个快速入门中，我们介绍了创建简单代理的基础知识，并逐步增强了其记忆能力。我们学习了如何传入聊天历史记录以及如何使用 `AIMessage` 和 `HumanMessage` 结构化消息。

自己试试吧！你可以更深入地研究不同类型的代理，尝试各种提示词，并集成其他工具来扩展代理的能力。

## LangChain v0.1 和 v0.2 代理之间的区别

当你进一步探索 LangChain 代理时，了解两个主要版本（v0.1 和 v0.2）之间的区别非常重要。

在 LangChain v0.1 中，代理被引入作为构建 AI 应用的强大工具。它们为创建动态且适应性强的代理奠定了基础，这些代理可以与各种工具交互并根据用户查询生成响应。然而，随着库的发展，LangChain 团队在 v0.2 中对代理框架进行了重大增强。

让我们仔细看看 LangChain v0.1 和 v0.2 代理之间的主要区别：

1. **简化的代理初始化**
   - 在 v0.1 中，初始化代理需要分别指定代理类型、工具、语言模型和其他参数。
   - 在 v0.2 中，该过程已简化。你现在可以使用 `initialize_agent` 函数，它会根据提供的工具和语言模型自动选择合适的代理类型。

2. **增强的代理类型**
   - LangChain v0.2 引入了新的和改进的代理类型，例如 `zero-shot-react-description` 代理，它为行动的自然语言描述提供了更好的支持。
   - `conversational-react-description` 代理已针对对话式 AI 应用进行了优化，以实现更无缝和连贯的对话。

3. **改进的工具集成**
   - v0.2 简化了将工具与代理集成的过程。`load_tools` 函数现在开箱即用地支持更广泛的工具，使得扩展代理的能力更加容易。
   - 可以通过继承 `BaseTool` 类并定义必要的方法来创建自定义工具，为整合特定领域的功能提供了更大的灵活性。

4. **增强的错误处理和调试**
   - LangChain v0.2 引入了更好的错误处理机制，使得诊断和修复代理实现中的问题更加容易。
   - `initialize_agent` 函数中的 `verbose` 参数允许你启用详细日志记录，以了解代理的决策过程并识别潜在问题。

### LangChain v0.2 中简化的代理初始化

以下是一个说明性示例，展示了 v0.2 中简化的代理初始化：

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
```



