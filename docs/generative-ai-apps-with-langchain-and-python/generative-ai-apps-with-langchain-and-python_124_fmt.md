# 第 8 章：你的第一个智能体应用

```python
user_prompt = "What is the revenue increase due to the benefits of AI"
agent.run(prompt_template.format(topic=user_prompt))
```

在此示例中，你使用必要的工具和一个提示模板来初始化智能体。用户提供一个提示，例如“What is the revenue increase due to the benefits of AI”，然后智能体使用该提示模板生成一篇关于给定主题的引人入胜的文章。

该智能体足够智能，能够利用已加载的适当工具，例如用于检索相关信息的`SerpAPI`和用于执行任何必要计算的`LLM-Math`，从而生成全面且准确的内容。

## 智能体作为任务管理器

你可以将智能体视为 AI 任务的项目经理。你给它们一个高层次的目标，它们会将其分解为可管理的步骤，将任务委派给适当的工具（如语言模型或外部 API），并协调整个流程。它们会分析结果，根据反馈做出决策，甚至从经验中学习以改进未来的表现。

## 实际应用示例

1.  **代码生成**：与其简单地要求语言模型“编写 Python 代码来计算斐波那契数列”，智能体可以更进一步。它可以：

    -   **优化你的请求**：“编写高效的 Python 代码，计算斐波那契数列直到第 50 个数。”

    -   **测试代码**：执行生成的代码并检查其正确性。

    -   **优化**：如果发现错误，提示模型进行修复。

    -   **解释**：清晰地解释代码的工作原理。

2.  **创意写作**：假设你想要一个关于时间旅行侦探的故事。智能体可以：

    -   将情节分解为多个场景。

    -   使用语言模型生成每个场景。

    -   确保角色发展和情节点的连贯性。

    -   修改和完善叙事，使其连贯且风格统一。

本质上，智能体通过以下方式增强了生成式 AI 模型的能力：

-   **扩展范围**：超越基本的文本或图像生成，处理复杂的多步骤任务。

-   **增加实用性**：专注于现实世界的成果和解决问题。

-   **提供自主性**：做出决策并适应不断变化的情况。

## 链与智能体有何不同？

既然你已经对智能体有了扎实的理解，你可能会想知道它们与链有何不同。

将链视为一系列按预定义顺序执行的连接步骤或任务。链中的每一步都将上一步的输出作为其输入，并对其进行处理以产生下一步的输出。链非常适合那些需要执行固定操作序列的场景。

以下是一个链的简单示例。请注意，此代码仅用于说明目的：

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# 定义提示模板
prompt_template = PromptTemplate(
    input_variables=["product"],
    template="What are the benefits of {product}?",
)

# 初始化 LLM
llm = OpenAI(temperature=0.9)

# 创建链
chain = LLMChain(llm=llm, prompt=prompt_template)

# 运行链
product_name = "Customer Service Chatbot"
response = chain.run(product_name)
print(response)
```

在此示例中，链由单个步骤组成：使用语言模型（`OpenAI`）和提示模板生成给定产品的优势。该链将产品名称作为输入，应用提示模板，并将生成的响应作为输出。

## 选择你的方法

以下是一个快速指南，可帮助你决定何时使用链与智能体：

| 因素 | 链 | 智能体 |
|--------|--------|--------|
| **任务结构** | 定义明确、顺序执行 | 开放式，需要探索或决策 |
| **灵活性** | 局限于预定义的序列 | 适应新信息，并可从多个工具中选择 |
| **易用性** | 通常更易于设置和理解 | 需要更多配置，但功能更强大 |
| **示例用例** | 文本摘要、基于文档回答问题、数据提取 | 创意写作、代码生成、复杂研究任务、自主决策场景 |

那么，智能体和链之间的主要区别是什么？

-   **自主性**：智能体是自主的，而链则不是。智能体可以在没有直接人工干预的情况下做出决策并采取行动，而链则依赖人工输入来运行。

-   **目标导向**：智能体是目标导向的，这意味着它们旨在实现特定目标或完成任务。另一方面，链更加灵活，可用于广泛的应用。请注意，也存在探索性智能体，它们旨在探索环境并从中学习，通常没有预定义的目标。

-   **上下文理解**：智能体能够理解其运行所处的上下文，从而生成更有意义和更相关的响应。链虽然能够处理上下文，但并非像智能体那样旨在理解上下文。

在智能体的世界中，语言模型作为决策者占据中心位置，根据给定的上下文确定要采取的行动序列。

可以这样理解：在链中，行动序列是预定义且硬编码的。这就像一步一步地遵循食谱，没有偏离的余地。但对于智能体，语言模型充当推理引擎，动态地选择要采取的行动及其顺序。它能够根据工具和期望的结果即兴发挥并调整步骤。

## 你的第一个端到端可工作的智能体应用

现在，让我们谈谈构建你的第一个智能体。我们已经讨论了智能体背后的代码，但这里是一个端到端完全可工作的智能体：

```python
# 安装所需的包
!pip install langchain==0.0.153
!pip install openai==0.27.6
!pip install python-dotenv==1.0.0
!pip install google-search-results==2.4.2

import os
from dotenv import load_dotenv
import openai
from langchain.agents import load_tools, initialize_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# 从 .env 文件加载环境变量
load_dotenv()

# 从环境变量获取 OpenAI API 密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 导入新的聊天补全 API：
os.environ["OPENAI_API_KEY"] = "your open AI key"
os.environ["SERPAPI_API_KEY"] = "your SerpAPI key"

# 初始化 OpenAI 客户端
openai.api_key = OPENAI_API_KEY

# 确认 API 密钥设置正确
openai.api_key = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# 初始化 ChatOpenAI 模型
chat_model = ChatOpenAI(model_name="gpt-4", temperature=0)

# 加载必要的工具
tools = load_tools(["serpapi", "llm-math"], llm=chat_model)

### 初始化智能体
agent = initialize_agent(tools, chat_model, agent="zero-shot-react-description", verbose=True)
```