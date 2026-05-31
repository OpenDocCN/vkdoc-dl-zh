# 第 1 章：LangChain 与大型语言模型入门

欢迎来到 LangChain 和大型语言模型的世界，在这里，你将学习如何使用最流行的生成式人工智能应用开发框架之一——LangChain——来构建生成式人工智能应用。



## LangChain 与 LLM 简介

你将学习如何利用这些能力超强的大型语言模型（我们常称之为 `LLM`）所蕴含的广博知识。我们将一同探索如何通过 `LangChain` 调用像 `GPT-4`、`PaLM` 和 `Gemini` 这样强大的 `LLM`，来开发一些令人惊叹、智能且贴近真实世界的应用，这些应用甚至能展现出类人的交互感。

`LangChain` 的强大之处在于，它让我们能够轻松利用大型语言模型（`LLM`）的能力来构建实际应用。无论你是经验丰富的程序员，还是刚刚入门的新手，你都会发现 `LangChain` 出奇地易用。正是这种编码的便捷性让我对 `LangChain` 着迷。我希望，当你开始通过本书中的实践示例学习，并发现它有多么简单时，你也会被它吸引。它的美妙之处在于，你甚至不需要成为机器学习大师或数据科学专家，就能利用它的强大功能。

© Rabi Jay 2024

R. Jay, *Generative AI Apps with LangChain and Python*,

[`doi.org/10.1007/979-8-8688-0882-1_1`](https://doi.org/10.1007/979-8-8688-0882-1_1#DOI)

## 第 1 章：LangChain 与 LLM 简介

在本章结束时，我相信你将掌握 `LangChain` 的精髓，并开始开发你自己的、由 `LLM` 驱动的生成式 AI 应用。

我希望你觉得这本书既全面又注重实践。我的目标是让你不仅理解 `LangChain` 和 `LLM` 的理论，还能将所学知识付诸实践，让你的生成式 AI 项目真正落地。

### 理解 LangChain

`LangChain` 是一个强大的框架，它能帮助你轻松开发基于 `LLM` 的人工智能应用。让我们来仔细看看。

`LangChain.com` 网站上对 `LangChain` 的官方定义如下：

> **LangChain** 是一个用于开发**由语言模型驱动的应用**的**框架**。它支持的应用能够：
> 
> - **感知上下文**：将语言模型连接到上下文来源（如提示指令、少量示例、用于支撑其回答的内容等）
> - **进行推理**：依赖语言模型进行推理（例如，如何根据提供的上下文进行回答、应采取哪些行动等）

`LangChain` 本质上是一个数字工具箱，你可以用它来构建令人惊叹的智能应用，这些应用能够在一定程度上像人类一样交谈、理解甚至思考。

以下是一些优势：

- 你可以利用像 `GPT-4`、`PaLM`、`Gemini` 这样的高级语言模型，甚至是 `LLaMA` 这样的开源模型所蕴含的广博知识。这为你所能开发的应用类型打开了无限可能。
- 你可以将这些 `LLM` 与你自己的特定私有数据集成。这意味着你可以根据你业务或项目的独特需求和背景，更精确地定制 `LLM` 的输出。
- 好消息是，你不受限于任何特定的 `LLM`，可以根据需要混合搭配不同的模型。这为生成式 AI 应用开发提供了高度的定制化能力，能够真正推动创新。

无论你是想开发提升客户服务的聊天机器人、生成创意内容的系统，还是自动化重复任务的解决方案，`LangChain` 都能提供你所需的工具。看到我们能够以多种方式应用这项技术来解决现实世界的问题并推动各行各业的创新，这非常鼓舞人心。

### 使用 LangChain 构建一个简单的生成式应用

下面是一个简单的示例，演示了一个使用 `GPT-4` 根据用户输入生成创意内容的基础应用。这看起来可能是一个微不足道的例子，但我分享它只是为了说明调用 `LLM` 是多么简单。我们将在后续章节中深入探讨这些主题。现在，只需快速浏览一下，以便对我们即将学习的内容有一个大致的了解。

```
#### 安装 LangChain 和 OpenAI 模块
!pip install openai==0.28.0
!pip install LangChain==0.1.20
```



#### 导入 LangChain 和 OpenAI 库

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
```

#### 使用 LangChain 初始化 OpenAI 模型

```python
LLM_OPENAI_API_KEY = "your_openai_api_key"
llm = OpenAI(api_key=LLM_OPENAI_API_KEY)
```

#### 定义用于生成故事创意的提示模板

```python
prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="基于以下行业输入生成一个创意产品构想：{user_input}",
)
```

#### 用户输入主题

```python
user_input = "环保家用电器"
```

