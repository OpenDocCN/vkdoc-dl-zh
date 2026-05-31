# 现在，你可以开始为你的应用添加更多功能了！

- **记忆（Memory）**  
  你可以使用此组件在链的多次运行之间持久化应用状态，并在不同会话中保持连续性和上下文。

- **回调（Callbacks）**  
  你可以使用回调来记录和流式输出任何链的中间步骤，从而在执行过程中提供透明度和可追溯性。

## 第 3 章 构建问答与聊天机器人应用

如你所见，这些组件中的每一个都在基于 LangChain 的应用的功能和效率中扮演着至关重要的角色。在接下来的几章中，你将学习如何使用它们来构建 LLM 应用。

### 生产阶段

在生产阶段，你的重点是确保应用程序平稳高效地运行。你可以使用 LangChain 生态系统中的平台 LangSmith 来检查、监控和评估应用程序的性能。

以下是如何使用 LangSmith 检查应用程序的一些见解：

```python
import os

!pip install langchain==0.2.7 langchain_openai==0.1.16

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.smith import RunEvalConfig, run_on_dataset

# 设置你的 API 密钥
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_api_key"

# 初始化你的模型和链
llm = ChatOpenAI(temperature=0)
prompt = PromptTemplate.from_template("讲一个关于 {topic} 的简短笑话")
chain = LLMChain(llm=llm, prompt=prompt)

# 定义你的评估配置
eval_config = RunEvalConfig(
    evaluators=[
        "criteria",
        "embedding_distance",
    ],
    custom_evaluators=[],
)

# 在数据集上运行评估
results = run_on_dataset(
    client=client,
    dataset_name="my-dataset",
    llm_or_chain_factory=chain,
    evaluation=eval_config,
)

# 打印结果
print(results)
```

在这个例子中，我们使用了 LangSmith，这是 LangChain 的评估和监控平台。它允许你在数据集上运行你的链，并使用各种指标评估其性能。`RunEvalConfig` 让你指定要使用的评估器，而 `run_on_dataset` 则执行评估。

**注意**：要使用 LangSmith，你需要注册一个账户并获取 API 密钥。请访问 [`smith.langchain.com/`](https://smith.langchain.com/) 开始使用。请记住对你的 API 密钥保密。

### 使用 LangServe 进行部署

在部署阶段，你将使用 LangServe 进行部署，通过将任何链轻松转换为 REST API，让你的 LLM 应用能够被其他服务通过网络访问。

要将你的 LangChain 应用部署为 API，LangServe 提供了一个简单的命令行界面。设置好应用程序后，你可以通过一个命令来启动服务：

```bash
poetry run langchain serve --port=8100
```

此命令将在 8100 端口启动一个服务器，使你的 LangChain 应用可被访问。

此命令使用了 LangServe，这是一个旨在轻松部署 LangChain 应用的工具。它会自动为你的链和代理设置必要的 API 端点，使其可通过 HTTP 访问。

在运行此命令之前，请确保你的项目中已安装 LangServe：

```bash
poetry add langserve
```

同时，确保你的项目结构遵循 LangServe 的约定。通常，这意味着需要有一个 `server.py` 文件来定义你的链以及它们应如何被服务。

## LangChain 生态系统

LangChain 最初只是一个简单的 Python 包，但在活跃开发者社区的投入和协作下，已发展成为一个强大的框架。随着它的发展，LangChain 团队意识到需要简化架构以提高可用性和可扩展性。

在开发应用程序时，你需要理解这些组件才能有效地使用它们并避免潜在的混淆。以下是它们是如何组织各个部分的。

### LangChain-Core

LangChain-Core 是该框架的基础。

- 它提供了核心抽象，这些抽象已成为 LangChain 组件的标准构建块。
- 你可以使用 LangChain 表达式语言来流畅地组合这些组件。
- 目前版本为 0.1，LangChain-Core 确保任何重大更新都会附带次版本号的提升，以保持对你的稳定性。

以下是使用 LangChain Core 的示例：

```python
# 使用 LangChain Core 的示例
from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI

# 初始化你的语言模型
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 使用模型
response = llm.invoke("讲一个关于编程的笑话。")
print(response.content)
```

此示例演示了如何使用 LangChain Core 与 OpenAI 聊天模型进行交互。`ChatOpenAI` 类提供了一个用于处理 OpenAI GPT 模型的高级接口。`langchain_core.language_models` 中的 `BaseLLM` 类定义了 LangChain 中所有语言模型都应实现的抽象接口。

LangChain 的结构和最佳实践在不断演进。请始终参考 [`python.langchain.com/`](https://python.langchain.com/) 上的最新文档，以获取关于类和用法模式的最新信息。

### LangChain-Community

LangChain-Community 是一个包含第三方集成的包。

- 它简化了你与外部数据源和工具连接的过程。
- 随着更多合作伙伴关系和协作的形成，你可以期待这个包会不断扩展。

**使用示例**

以下是其使用方式的示例：

```python
# 利用第三方集成的示例
from langchain_community.document_loaders import TwitterTweetLoader

# 初始化 Twitter 加载器
loader = TwitterTweetLoader(
    query="LangChain",
    bearer_token="your_twitter_bearer_token",
    num_tweets=100
)

# 加载推文
documents = loader.load()
```

此示例演示了如何使用 LangChain 社区贡献的 Twitter 加载器来获取推文。`TwitterTweetLoader` 允许你根据查询搜索推文，并将其作为文档加载，以便在 LangChain 管道中进行进一步处理。

请注意，你需要一个 Twitter 开发者账户和一个 bearer token 才能使用此加载器。同时，在你的应用程序中使用此加载器时，请注意 Twitter 的 API 使用限制和服务条款。

请始终参考最新的 LangChain 文档，以获取关于可用加载器及其用法的最新信息：[`python.langchain.com/docs/integrations/document_loaders/`](https://python.langchain.com/docs/integrations/document_loaders/)。

### 高级组件

**LangChain**：LangChain 包包含高级的、针对特定用例的链、代理和检索算法，它们构成了你的生成式 AI 应用架构的骨干。该包即将发布稳定的 0.1 版本，将为你带来构建复杂 AI 驱动解决方案的复杂功能。

以下是一个说明性示例：



