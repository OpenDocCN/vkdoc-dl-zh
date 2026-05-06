# 输出：`'Page: LangChain\nSummary: LangChain is a framework designed to simplify the creation of applications '`

您刚刚学习了如何使用内置工具并根据自己的喜好进行自定义。LangChain 提供了丰富的资源来帮助您继续探索：

- **内置工具**：请查阅官方文档，获取所有内置工具的完整列表。
- **自定义工具**：虽然内置工具很方便，但您很可能需要针对特定用例定义自己的工具。LangChain 提供了关于如何创建自定义工具的指南。
- **工具包**：工具包是能够协同工作的工具集合。文档提供了所有内置工具包的详细描述和列表。
- **作为 OpenAI 函数的工具**：LangChain 中的工具与 OpenAI 函数类似，您可以轻松地将它们转换为该格式。请查阅官方笔记本以获取相关操作说明。

#### 工具包

工具包是经过精心策划的工具集合，旨在为特定任务无缝协作。有时，完成一项任务需要一组相关工具协同工作。这时工具包就派上了用场。它们带有便捷的加载方法，使您可以更轻松地开始使用所需的工具。例如，GitHub 工具包包含用于搜索 GitHub 问题、读取文件、评论问题等的工具。LangChain 提供了现成工具包的完整列表，您可以在文档的“集成”部分找到它们。

首先，您需要初始化要使用的工具包。假设您正在使用 `ExampleToolkit`（目前只是一个占位符）：`toolkit = ExampleToolkit(...)`

每个工具包都公开了一个 `get_tools` 方法，该方法返回该工具包中包含的工具列表。以下是访问它们的方法：`tools = toolkit.get_tools()`

现在，有了这些工具，您可以创建一个能够利用它们集体力量的代理。LangChain 提供了一个 `create_agent_method` 函数，允许您实现这一点。只需传入您的 LLM（大型语言模型）、工具列表和提示（如果需要），您就拥有了一个随时可以处理任何任务的代理，如下所示：`agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)`

有了这个代理的指挥，您可以轻松地编排这些工具来执行复杂的任务、简化工作流程，并取得原本可能具有挑战性或耗时的成果。

LangChain 文档的“集成”部分提供了现成工具包的完整列表。您拥有从网页抓取、数据库操作、社交媒体平台、数据处理到自然语言处理等各种工具包。通过组合正确的工具，您可以轻松创建强大的工作流程并自动化复杂的任务。

### 注意事项

在使用工具时，有两个重要的设计注意事项需要牢记：

1.  **为代理提供正确的工具**：为您的代理配备实现目标所必需的工具至关重要。如果没有正确的工具集，您的代理的能力将受到限制，并且可能难以完成手头的任务。
2.  **以对代理最有帮助的方式描述工具**：您描述工具的方式对于代理能否有效使用它们起着至关重要的作用。您应该记住提供清晰的描述，解释每个工具的用途，以便代理能够就何时以及如何使用它们做出明智的决定。

## 使用 LangGraph 构建增强能力的代理

让我们看看如何使用 LangGraph 通过利用图结构来管理和处理信息，从而创建更复杂、更有能力、响应更快的代理。

### 什么是 LangGraph？

LangGraph 允许您以图格式构建信息，其中节点代表数据或任务片段，边代表它们之间的关系。这种结构使代理更容易处理复杂的工作流程、理解上下文并高效地执行多步骤任务。

想象一下，您有一个代理需要根据用户输入执行一系列任务。如果没有结构化的方式来管理这些任务，您的代码可能会变得混乱且难以维护。LangGraph 通过以既合乎逻辑又可扩展的方式组织任务和数据来帮助您。

### 设置 LangGraph

首先，让我们设置您的环境。确保您已安装 LangChain：

1.  **安装 LangChain**：如果尚未安装，请使用 pip 安装 LangChain。

```
!pip install langchain==0.2.5 langchain_openai==0.1.8 langgraph==0.1.8
```

2.  **导入必要模块**：您将从 LangChain 导入所需的类。

```
from langchain_openai import OpenAI
from langchain.agents import Agent
from langchain.graph import LangGraph, Node
```

3.  **初始化语言模型**：使用 OpenAI API 密钥设置您的语言模型。

```
llm = OpenAI(api_key="your_openai_api_key")
```

### 创建一个简单的 LangGraph

让我们创建一个简单的 LangGraph，它代表代理的任务序列。假设您正在构建一个帮助用户规划旅行的旅行代理：

1.  **定义节点**：节点是图中的单个任务或信息片段。

```
# 定义节点
node1 = Node(name="GreetUser", action=lambda: "您好！今天我能如何协助您规划旅行？")
node2 = Node(name="GetDestination", action=lambda user_input: f"好选择！{user_input} 听起来是一个绝佳的目的地。")
node3 = Node(name="SuggestActivities", action=lambda: "以下是一些您可能喜欢的活动：参观博物馆、品尝当地美食以及探索自然小径。")
```

2.  **创建图**：将节点链接起来形成工作流程。

```
# 创建 LangGraph
travel_graph = LangGraph()
travel_graph.add_node(node1)
travel_graph.add_node(node2)
travel_graph.add_node(node3)

# 定义边（节点之间的关系）
travel_graph.add_edge("GreetUser", "GetDestination")
travel_graph.add_edge("GetDestination", "SuggestActivities")
```

3.  **定义代理**：创建一个使用此图与用户交互的代理。

```
class TravelAgent(Agent):
    def __init__(self, llm, graph):
        super().__init__(llm=llm)
        self.graph = graph

    def run(self, user_input):
        # 从第一个节点开始，并遍历图
        current_node = self.graph.get_node("GreetUser")
        response = current_node.action()
        print(response)

        current_node = self.graph.get_node("GetDestination")
        response = current_node.action(user_input)
        print(response)

        current_node = self.graph.get_node("SuggestActivities")
        response = current_node.action()
        print(response)

# 实例化代理
agent = TravelAgent(llm=llm, graph=travel_graph)
```



