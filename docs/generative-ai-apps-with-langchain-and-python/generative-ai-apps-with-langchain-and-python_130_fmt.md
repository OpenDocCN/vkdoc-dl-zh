# 使用最终输出创建 `AgentFinish`

`final_output = AgentFinish(return_values={"output": "The annual revenue of Amazon is 574.8 billion."})`

在此示例中，你创建了一个 `AgentAction`，指定了要使用的工具（例如 "search"）以及提供给该工具的输入。然后执行该动作并存储输出。由该动作及其输出组成的中间步骤被添加到 `intermediate_steps` 列表中。最后，你使用最终输出创建了一个 `AgentFinish`，在本例中，该输出是一个声明亚马逊年收入的字符串。

你可以看到，智能体能够利用这些大语言模型和工具的强大功能，动态地进行推理，并确定解决给定问题的最佳行动方案。它们能够根据可用的工具、提供的输入以及在此过程中获得的中间结果进行调整和决策。

### 智能体

好了，让我们来谈谈智能体的核心，即负责决定下一步行动的链。它通常由语言模型、提示词和输出解析器驱动。

你应该记住，不同的智能体在推理、编码输入和解析输出方面都有自己独特的风格。

LangChain 提供了多种内置智能体供你选择。每个智能体都有自己的优势和特点。你可以在本章“延伸阅读”部分包含的智能体类型文档中找到这些智能体的完整列表。

如果你需要更多控制权或有特定要求，你可以轻松构建自己的自定义智能体。构建自定义智能体允许你定义自己的提示风格、输入编码和输出解析逻辑。我们将在本章后面讨论这一点。

让我们深入了解智能体的输入和输出。

## 第 9 章 构建不同类型的智能体

### 智能体输入

谈到智能体的输入，关键就在于键值对。唯一必需的键是 `intermediate_steps`，它对应于我们之前讨论过的中间步骤。这些步骤至关重要，因为它们为智能体提供了到目前为止已完成操作的上下文。

但这里就是 `PromptTemplate` 发挥作用的地方。它负责将这些键值对转换为语言模型易于理解的格式。

### 智能体输出

接下来是智能体输出。智能体的输出可以是下一步要执行的动作，也可以是返回给用户的最终响应。用技术术语来说，这些输出由 `AgentAction` 或 `AgentFinish` 表示。你可以将它们视为智能体的决策或最终裁决。

输出可以是以下三种类型之一：

- `AgentAction`：智能体下一步想要执行的单个动作

- `List[AgentAction]`：智能体下一步想要执行的一系列动作

- `AgentFinish`：智能体想要返回给用户的最终响应

这就像智能体在说明下一步或答案是什么。输出解析器负责接收来自语言模型的原始输出，并将其转换为这三种类型之一。换句话说，它解释智能体的想法，并将其转化为具体的动作或响应。

## 第 9 章 构建不同类型的智能体

以下是一个代码片段，用于说明智能体输入和输出的用法：

```python
from langchain_core.agents import AgentAction, AgentFinish

# 使用新方法创建智能体
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# 定义智能体的输入
agent_input = {
    "intermediate_steps": [
        (AgentAction(tool="Search", tool_input="What is the capital of France?", log="Searching for the capital of France"), "Paris is the capital of France.")
    ]
}

# 创建一个 AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 使用输入调用智能体
agent_output = agent.run(agent_input)
```