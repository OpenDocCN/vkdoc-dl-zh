# 检查智能体输出的类型

```
if isinstance(agent_output, AgentAction):
    print("智能体想要执行以下操作：", agent_output)
elif isinstance(agent_output, list):
    print("智能体想要执行以下操作：", agent_output)
elif isinstance(agent_output, AgentFinish):
    print("智能体已完成，响应如下：", agent_output)
```

## 第 9 章 构建不同类型的智能体

在此示例中，你通过提供 `intermediate_steps` 来定义智能体的输入，这是一个包含先前操作及其输出的元组列表。然后，你使用此输入调用智能体，并将输出存储在 `agent_output` 中。

最后，你使用 `isinstance()` 检查智能体输出的类型，以确定它是 `AgentAction`、`AgentAction` 列表还是 `AgentFinish`。根据类型，你可以采取相应的操作或将最终响应发送回用户。

现在，你已经对智能体如何做出决策、执行操作以及提供响应有了扎实的理解。

### `AgentExecutor`

`AgentExecutor` 是幕后的核心引擎，它为智能体平稳高效地运行提供运行时环境。它负责调用智能体、执行智能体选择的操作、将操作输出传回给智能体，并重复此过程，直到智能体得出结论。这就像智能体与执行器之间的通信循环，执行器促进信息和操作的流动。

以下是 `AgentExecutor` 工作原理的简化伪代码表示，同样仅用于说明目的：

```
next_action = agent.get_action(...)
while next_action != AgentFinish:
    observation = run(next_action)
    next_action = agent.get_action(..., next_action, observation)
return next_action
```

这看起来可能很简单，但 `AgentExecutor` 在幕后处理了多项复杂任务，以简化你的工作。让我们回顾一些场景：

1. 当智能体选择了一个不存在的工具时，执行器会优雅地处理这种情况，并使智能体保持在正轨上。
2. 如果工具在执行过程中遇到错误，执行器会捕获异常并进行适当管理，以确保智能体能够继续其工作。
3. 当智能体产生的输出无法解析为有效的工具调用时，执行器会处理这种情况，并引导智能体回到有效路径。
4. 执行器在所有级别提供全面的日志记录和可观测性，例如智能体决策和工具调用。它可以将此信息输出到 `stdout` 和/或发送到 LangSmith 以进行进一步分析和可视化。

### 工具与工具包

让我们快速了解一下工具和工具包。

#### 工具

工具是智能体、链或 LLM（大型语言模型）可以用来与世界交互的接口。它们结合了几个基本要素：

1. **工具名称**：一个简洁、描述性的标签，告诉你该工具的功能。
2. **描述**：对工具用途和功能的简要说明。
3. **JSON 模式**：工具所需输入的结构化定义。可以将其视为如何正确使用工具的蓝图。
4. **要调用的函数**：执行工具操作的实际代码。
5. **标志**：一个标志，用于确定工具的输出是立即可见还是需要进一步处理。

名称、描述和 JSON 模式帮助 LLM 理解如何指定所需的操作，而要调用的函数则相当于实际执行该操作。

LangChain 中的工具抽象由两个关键组件组成：

1. **工具的输入模式**：这就像一个蓝图，告诉语言模型（LLM）调用该工具需要哪些参数。提供命名合理且描述清晰的参数至关重要，这样 LLM 在调用工具时就能确切知道要提供哪些输入。
2. **要运行的函数**：这是工具被调用时实际执行的 Python 函数。它是根据提供的输入执行所需操作的代码。

你需要记住的一个重要点是，工具的输入越简单，LLM 就越容易使用它。我建议你使用具有单个字符串输入的工具，因为智能体与它们配合得很好。

LangChain 有文档说明哪些智能体类型可以处理更复杂的输入。请参阅“延伸阅读”部分以获取文档链接。

现在，让我们使用 `WikipediaQueryRun` 工具，它是维基百科的一个便捷包装器：

```
!pip install langchain==0.2.5 langchain_openai==0.2.5 wikipedia
```

```
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# 使用自定义配置初始化工具
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)
```

现在，让我们探索该工具的一些属性：

```
# 检查默认名称
print(tool.name)  # 输出：'Wikipedia'

# 检查默认描述
print(tool.description)  # 输出：'A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.'

# 检查输入的默认 JSON 模式
print(tool.args)  # 输出：{'query': {'title': 'Query', 'type': 'string'}}

# 检查工具是否应直接返回给用户
print(tool.return_direct)  # 输出：False
```

让我们尝试搜索关于 LangChain 本身的信息：

```
# 使用字典输入调用工具
print(tool.run({"query": "langchain"}))
# 输出：'Page: LangChain\nSummary: LangChain is a framework designed to simplify the creation of applications '

# 使用单个字符串输入调用工具（因为它只期望一个输入）
print(tool.run("langchain"))
# 输出：'Page: LangChain\nSummary: LangChain is a framework designed to simplify the creation of applications '
```

但是，如果我们想要自定义工具的名称、描述或 JSON 模式呢？让我们为维基百科工具创建一个自定义模式：

```
from langchain_core.pydantic_v1 import BaseModel, Field

class WikiInputs(BaseModel):
    """维基百科工具的输入。"""
    query: str = Field(description="要在维基百科中查询的查询词，应为 3 个或更少的词")

# 现在，让我们使用自定义设置创建工具的新实例
tool = WikipediaQueryRun(
    name="wiki-tool",
    description="在维基百科中查找内容",
    args_schema=WikiInputs,
    api_wrapper=api_wrapper,
    return_direct=True,
)

# 检查更新后的属性
print(tool.name)  # 输出：'wiki-tool'
print(tool.description)  # 输出：'在维基百科中查找内容'
print(tool.args)  # 输出：{'query': {'title': 'Query', 'description': '要在维基百科中查询的查询词，应为 3 个或更少的词', 'type': 'string'}}
print(tool.return_direct)  # 输出：True

# 使用更新后的设置使用工具
print(tool.run("langchain"))
```



