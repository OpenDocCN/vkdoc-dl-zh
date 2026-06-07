# 第 2 章：使用 LangChain 集成 LLM API

还有其他保护 API 密钥的方法，例如：

- 使用 AWS Secrets Manager、HashiCorp Vault 或 Azure Key Vault 等密钥管理工具来存储密钥
- 将密钥作为环境变量存储在服务器上
- 在应用程序和 OpenAI API 之间设置代理服务器来存储 API 密钥，然后你的应用程序将调用代理而非 OpenAI API

**注意**：切记不要将你的 API 密钥提交或发布到公共仓库，也不要与未经授权的第三方共享。

**恭喜！**

你已经成功设置了开发环境并将其连接到 OpenAI API。现在，你已经准备好探索 LangChain 的精彩世界，并开始使用大型语言模型构建强大的应用程序。

请为自己鼓掌，因为你已经迈出了成为 LangChain 大师的第一步，也是非常重要的一步！

## 练习 1：直接调用 LLM API

在揭示 LangChain 的强大之处之前，让我们快速了解一下直接调用 OpenAI API 与使用 LangChain 框架之间的区别。这看起来可能是一个简单的例子，但你会开始看到使用 LangChain 的 API 来抽象化 LLM API 调用的好处。通过这种方法，你可以采用更即插即用的方式，并避免跨多个提供商进行直接 LLM API 调用所带来的复杂性。

下面是使用直接 LLM API 与 LangChain 设置基本文本生成任务的代码。

### 直接 LLM API 方法

**任务**：使用 OpenAI API 根据给定的提示生成一个短篇故事。

**步骤 1：选择 LLM API 提供商并获取 API 密钥**

前往 OpenAI 网站并注册一个账户。导航到 API 部分，按照说明获取你的 API 密钥。此密钥将允许你的应用程序向 OpenAI 的服务器进行身份验证并向 GPT-4 发出请求。

**步骤 2：安装必要的 SDK 或库**

OpenAI 提供了一个官方的 Python 库，可以简化与 GPT-4 的交互。你可以使用 Python 的包管理器 `pip` 来安装这个库。打开你喜欢的代码编辑器，创建一个新的 Python 文件。

在这个文件中，你将编写一个脚本，提示用户输入一个故事开头，然后将这个提示发送给 GPT-4，由它生成后续内容。

这是一个简单的例子：

```python
# 安装 openai 包版本 0.28
!pip install openai==0.28

import os
import openai

# 从 openai 导入新的 Chat Completion API
import ChatCompletion

# 使用环境变量设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = "sk-T"

# 确认 API 密钥设置正确
openai.api_key = os.getenv("OPENAI_API_KEY")

# 定义一个函数来获取聊天补全
def get_chat_completion(user_prompt):
    # 使用 Chat Completion API 生成响应
    response = ChatCompletion.create(
        # 指定要使用的聊天模型引擎
        model="gpt-3.5-turbo",
        # 将用户提示作为消息提供
        messages=[{"role": "user", "content": user_prompt}]
    )
    # 提取并返回生成的响应
    return response.choices[0].message.content.strip()

# 提示用户输入故事提示
user_prompt = input("输入一个故事提示：")

# 根据用户提示生成聊天补全
result = get_chat_completion(user_prompt)

# 打印生成的结果
print(result)
```

**注意**：请记住导入正确版本的 OpenAI 库，否则将无法工作。

让我们分解一下代码中发生的事情：

- 我们安装版本为 0.28 的 OpenAI 包。
- 我们导入必要的模块：`os` 和 `openai`。
- 我们从 OpenAI 模块导入新的 Chat Completion API。
- 我们使用环境变量设置 OpenAI API 密钥。
- 我们确认 API 密钥设置正确。
- 我们定义 `get_chat_completion` 函数，使用 Chat Completion API 生成聊天补全。它接受一个 `user_prompt` 作为输入，并将其作为消息发送给 API。`model` 参数指定要使用的聊天模型引擎（例如 `"gpt-3.5-turbo"`）。
- 从 API 响应中提取生成的响应并返回。

代码会提示用户输入一个故事提示，如图 2-1 所示。

***图 2-1.** 输入故事提示*

输入你的内容，在本例中，我输入了“讲述地球的故事”，然后按回车。`get_chat_completion` 函数被调用，使用用户提示生成聊天补全。

生成的结果会被打印出来，如图 2-2 所示。

***图 2-2.** LLM 对提示的响应*

**成果**

完成此练习后，你将获得以下实践经验：

- 获取并使用 API 密钥向 LLM API 进行身份验证
- 安装并使用 Python SDK 简化 API 交互
- 向 LLM（GPT-4）发出请求并处理其响应

## 练习 2：使用 LangChain 增强灵活性

现在你已经了解了如何直接调用 LLM API，让我们探索 LangChain 如何通过使代码更灵活来简化事情。

**步骤 1：在开发环境中设置 LangChain**

要开始使用 LangChain，你需要通过 `pip` 安装它。打开你的终端或命令提示符或任何其他开发工具，例如 Google Colab，并执行以下命令：

```bash
pip install langchain
```

此命令安装 LangChain 及其依赖项，为你的开发做好准备。

**步骤 2：修改脚本以使用 LangChain**

打开你在练习 1 中创建的 Python 脚本。你将修改此脚本以使用 LangChain，而不是直接与 OpenAI API 交互。

在脚本开头导入 LangChain，并将其配置为使用 LLM（例如 OpenAI 的 GPT-4）。以下是修改脚本的示例：

```python
# 导入必要的模块
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 你的 OpenAI API 密钥
api_key = "your_api_key_here"

# 确认 API 密钥设置正确
openai.api_key = os.getenv("OPENAI_API_KEY")

# 使用 OpenAI 的 GPT 初始化 LangChain
llm = OpenAI(api_key=api_key)

# 使用 LLMChain 进行简单交互
chain = LLMChain(llm=llm)

# 定义用于生成响应的提示模板。
prompt_template = PromptTemplate(input_variables=["user_input"], template="你是一个有用的聊天机器人。用户：{user_input} 响应：")

# 创建 OpenAI 类的实例并赋值给变量 llm
llm = OpenAI()

# 创建 LLMChain 类的实例
chain = LLMChain(llm=llm, prompt=prompt_template)

# 提示用户输入故事开头
user_prompt = input("输入一个故事提示：")

# 调用 LLMChain 实例的 run 方法
response = chain.run(user_prompt)

# 打印生成的响应
print(response)
```

**注意**：LangChain 正在积极开发中，其结构可能会发生变化。请始终参考官方 LangChain 文档以获取最新的导入语句和使用模式。

从上面的例子中需要注意几点。首先，这种方法的优势在于它使用 LangChain 来抽象 API 交互，使得在不大量修改代码的情况下，更容易在 LLM 之间切换或调整提示工程方法。

接下来，让我们讨论我们导入的三个模块：

- `langchain.llms` 中的 `OpenAI`：这是用于与 OpenAI 语言模型交互的类。
- `langchain.prompts` 中的 `PromptTemplate`：此类用于定义用于生成响应的提示模板。
- `langchain.chains` 中的 `LLMChain`：此类表示一个链，它将语言模型（LLM）与提示模板结合起来以生成响应。

以下是代码中发生的事情：

- 我们使用 `os.environ["OPENAI_API_KEY"]` 将 OpenAI API 密钥设置为环境变量。API 密钥的值被赋值给该变量。
- 通过使用 `os.getenv("OPENAI_API_KEY")` 将其赋值给 `openai.api_key`，确认 API 密钥设置正确。
- 我们创建一个 `PromptTemplate` 对象，参数如下：
  - `input_variables=["user_input"]`：这指定模板期望一个名为 `"user_input"` 的变量。
  - `template="你是一个有用的聊天机器人。用户：{user_input} 响应："`：这定义了提示的模板字符串。它包含花括号 `{}` 内的 `"user_input"` 变量，以指示用户输入将被插入的位置。
- 我们创建一个 `OpenAI` 类的实例并将其赋值给变量 `llm`。这表示将用于生成响应的 OpenAI 语言模型。
- 我们创建一个 `LLMChain` 类的实例，参数如下：
  - `llm=llm`：这指定要在链中使用的语言模型，即上一步中创建的 OpenAI 实例。
  - `prompt=prompt_template`：这指定要在链中使用的提示模板，即之前创建的 `PromptTemplate` 对象。
- 使用 `user_prompt` 提示用户输入他们的请求；在我的例子中，我询问了“讲述地球的故事”。
- 调用 `LLMChain` 实例的 `run` 方法，参数为变量 `user_input`。这将根据提供的提示和用户输入从语言模型生成响应。
- 使用 `print(response)` 打印生成的响应。

总之，你刚刚学习了如何使用 LangChain 调用 OpenAI 语言模型、创建提示模板，并根据用户输入生成响应。然后将响应打印到控制台。

该代码演示了如何使用 LangChain 与 OpenAI API 交互、定义提示模板，并以结构化和模块化的方式使用语言模型生成响应。

**步骤 3：在 LangChain 中尝试提示工程**

LangChain 促进了高级提示工程技术。你可以尝试不同的提示格式和参数，看看它们如何影响故事的创造力、连贯性和相关性。

例如，你可以调整 `temperature` 参数以使响应更可预测或更不可预测，或使用停止序列来控制生成内容的长度和结构。以下是实现该结果的示例：

```python
chain = LLMChain(
    llm=llm, prompt=prompt_template,
    llm_kwargs={
        "temperature": 0.7,
        "max_tokens": 100,
        "stop": ["\n"]
    }
)
```

通过尝试不同的提示格式和参数，你可以看到提示或参数的变化如何影响输出。这将帮助你理解如何从 LLM 获得最佳响应。

**成果**

你可以观察到，LangChain 不仅简化了在不同 LLM 之间切换的过程，而且还为优化提示以获得更高质量的 AI 生成内容开辟了可能性。

LangChain 代码也更简洁、更简单，并且允许插入多个 LLM。此外，它与 LLM 的 API 没有紧密耦合。随着我们转向更复杂的用例，例如使用提示工程定制响应、集成外部数据源等，这些好处将变得更加明显。

## 关键要点

让我们讨论一下我们目前学到的内容：

- **简化开发**：你了解到 LangChain 通过抽象直接 LLM API 调用的复杂性，简化了生成式 AI 应用程序的开发。你可以更多地关注应用程序逻辑，而不是 LLM API 的细微差别。
- **增强的灵活性和可扩展性**：你可以轻松集成和切换不同的 LLM，而无需进行大量的代码重构。这确保了应用程序能够以最小的技术债务与新兴的 AI 技术一起发展。
- **优化的提示工程**：你可以使用 LangChain 工具和模板进行提示工程，这可以减少实验所需的时间以及从 LLM 引出所需响应所需的技能。
- **成本与效率**：通过利用 LangChain 的模块化开发和智能 API 请求管理方法，你可以优化成本并提高应用程序性能，特别是对于复杂和高容量的 AI 任务。
- **全球可访问性**：通过利用基于云的 LLM API，你可以构建和部署具有全球覆盖范围和一致性能的 AI 驱动应用程序。
- **安全性与数据处理**：你可以维护强大的安全标准，并简化复杂数据集的处理，同时遵守行业标准的安全要求。
- **向创新过渡**：从直接使用 LLM API 转向 LangChain，为生成式 AI 应用程序开发中的创新开辟了新的可能性。

总之，你可以使用 LangChain 来简化开发过程，增强应用程序的灵活性和可扩展性，并在解决 LLM API 固有挑战的同时开辟创新可能性。

## 开始使用 LangChain 进行创作

恭喜！你已经了解了 LLM API 的强大功能，并发现了 LangChain 的变革潜力。现在，是时候将你的技能付诸实践，将你的想法变为现实了。

在下一章中，我们将更深入地探讨如何构建 LLM 应用程序。同时，如果你还没有开始，我建议你开始你的第一个 LangChain 项目。现在，你可以从一个简单的内容生成器开始，然后构建一个聊天机器人或数据驱动的应用程序。

通过探索 LangChain 的文档、尝试不同的 LLM 以及发现新的用例，继续你的学习之旅。

## 技术术语表

以下是本章中使用的技术术语的定义列表：

- **API（应用程序编程接口）**：一组规则和定义，允许软件程序相互通信，以集成外部服务或数据。
- **LLM（大型语言模型）**：基于大量训练数据集，用于理解和生成类似人类文本的高级 LLM 模型。
- **SDK（软件开发工具包）**：用于为特定平台或技术开发应用程序的软件工具、库和文档的集合。
- **分词**：将文本分解为更小的单元（标记），例如单词或短语，以供机器学习模型处理的过程。
- **规范化**：将文本转换为一致格式（例如，小写、去除标点符号）以提高机器学习模型性能的过程。
- **困惑度**：衡量语言模型预测文本样本好坏程度的指标。较低的困惑度表示更好的预测准确性。

## 延伸阅读

以下是帮助你巩固本章所涵盖主题的理解，并进一步探索 LLM API 世界的资源列表：

- **OpenAI API 指南**：OpenAI 提供了关于如何使用其 API 的广泛文档，包括提示工程和模型选择的最佳实践。[`platform.openai.com/docs/introduction`](https://platform.openai.com/docs/introduction)
- **Google Cloud AI 服务**：Google Cloud 提供了广泛的 AI 服务。他们的文档对于希望利用 Google AI 能力的开发者来说是一个信息宝库。[`cloud.google.com/ai-platform/docs`](https://cloud.google.com/ai-platform/docs)
- **AWS AI 服务**：AWS 提供了其 AI 服务的全面指南，非常适合希望将 AWS 机器学习工具集成到其应用程序中的开发者。[`docs.aws.amazon.com/machine-learning/`](https://docs.aws.amazon.com/machine-learning/)
- **《大型语言模型的设计模式》**：本文探讨了在应用程序开发中有效利用 LLM 的各种策略和设计模式。[`arxiv.org/abs/2004.13214`](https://arxiv.org/abs/2004.13214)