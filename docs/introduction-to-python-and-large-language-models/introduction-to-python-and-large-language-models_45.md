# 7. 利用 Python 3.11 和 Python 库进行大语言模型开发

在不断发展的人工智能领域，大语言模型已成为从自然语言处理到复杂人工智能驱动解决方案等多种应用的强大工具。Python 3.11 的出现带来了许多新特性和优化，显著增强了这些复杂模型的开发。结合一系列强大的 Python 库，本章深入探讨了如何有效利用 Python 3.11，借助 `LangChain`、`Hugging Face`、`Pinecone`、`OpenAI`、`Cohere` 和 `Lamini.ai` 等平台，开发基于大语言模型的应用。

我们探讨了各种类型的数据源以及数据收集策略的重要性。接着，我们将指导您完成数据清洗、标准化和转换的过程，强调每一步在消除噪声和提高数据质量方面的重要性。我们将特别关注分词和嵌入技术，这些技术对于将文本数据转换为大语言模型可以有效处理和理解的形式至关重要。

## LangChain

`LangChain` 是一个公共开源平台，旨在赋能从事人工智能和机器学习领域的开发者。它促进了大型语言模型与各种外部系统的集成，从而能够创建由大语言模型驱动的应用。`LangChain` 的主要目标是在强大的大语言模型（如 OpenAI 的 GPT-3.5 和 GPT-4、`Cohere`）与多个外部数据源之间建立连接。这种集成旨在增强自然语言处理应用的开发和利用。

该框架面向精通 Python、JavaScript 或 TypeScript 的开发者、软件工程师和数据科学家，提供这些语言的软件包。`LangChain` 由 Harrison Chase 和 Ankush Gola 于 2022 年作为公共开源项目发起，其第一个版本也在同年发布。

`LangChain` 的重要性在于它能够简化生成式人工智能应用的创建过程。它通过组织和使大量数据易于访问，为开发者提供了一条构建复杂自然语言处理应用的简化途径。这对于需要处理和访问海量数据集的大语言模型尤其有益。

### LangChain 特性

`LangChain` 包含一套旨在增强自然语言处理应用开发和功能的组件：

- **模型交互：** 这一方面，也称为模型输入/输出，促进了与任何语言模型的交互。它负责管理向模型输入数据以及解释输出数据。
- **数据连接与检索：** 此功能允许对大语言模型可访问的数据进行转换、存储和检索。数据可以存储在数据库中，并通过查询获取。
- **链：** 为了创建复杂的应用，`LangChain` 能够集成各种组件或多个大语言模型。此过程创建了所谓的 LLM 链，连接不同的模型和工具。
- **代理：** 通过代理模块，大语言模型可以确定解决问题的最佳行动方案。这是通过向大语言模型和其他资源发送一系列指令来实现的，引导它们完成特定请求。
- **记忆：** 此组件帮助大语言模型保留用户交互的上下文。它允许根据应用需求整合短期和长期记忆。

### LangChain 有哪些集成？

`LangChain` 还通过与 LLM 提供商和外部数据源的集成来支持应用。它可以通过合并来自 `Hugging Face`、`Cohere` 和 `OpenAI` 等实体的大语言模型与 `Apify Actors`、`Google Search` 和 `Wikipedia` 等数据存储库，来创建聊天机器人或问答系统。这种融合允许应用处理用户查询，并从这些平台获取最佳响应。

此外，`LangChain` 可以与云存储服务（如 Amazon Web Services、Google Cloud 和 Microsoft Azure）以及用于存储和查询高维数据的向量数据库（如 `Pinecone`）集成。这些集成利用尖端的自然语言处理技术来打造高效且有效的应用。

### 如何在 LangChain 中构建应用？

使用 `LangChain` 创建应用涉及利用语言模型的能力来构建定制化应用。

开发过程通常遵循几个基本步骤：

1.  **定义应用目的：** 首先，开发者需要确定应用的具体功能和范围。这包括确定将使用的必要集成、组件和语言模型。
2.  **开发应用逻辑：** 使用提示词，开发者可以构建应用将遵循的逻辑或功能。
3.  **功能定制：** `LangChain` 为开发者提供了调整和修改其代码的灵活性，从而能够创建满足应用特定需求的定制功能。
4.  **优化语言模型：** 为任务选择合适的语言模型，并根据应用的具体需求对其进行调优，对于获得最佳性能至关重要。
5.  **数据准备：** 通过清洗技术确保数据的清洁和准确性至关重要，同时还要实施安全协议以保护敏感信息。
6.  **持续测试：** 为保持应用的效率和可靠性，需要进行持续的测试。

这种方法能够开发出利用语言模型能力来满足多样化和特定用例的健壮应用。



### LangChain 的应用场景

LangChain 在利用大型语言模型（LLM）方面的能力，为众多行业和领域解锁了广泛的高级应用。以下是一些示例和应用场景：

- **客户支持聊天机器人：** LangChain 有助于创建能够处理复杂咨询甚至执行交易的先进聊天机器人。这些机器人旨在理解和记住用户对话的上下文，类似于 ChatGPT 的运作方式，从而提升客户服务与体验。

- **编程助手：** 利用 LangChain，结合 OpenAI 等平台的 API，可以开发出帮助软件开发者和技术专业人士提升编码能力、提高生产力的工具。

- **医疗健康创新：** 在医疗领域，基于 LangChain 构建的应用正在革新诊断方式，并简化预约安排等行政任务。这种自动化让医疗专业人员能够将更多时间投入到关键的患者护理中。

- **营销与电商工具：** 由 LLM 驱动的应用通过理解消费者行为、购买模式和产品细节，正在改变电商和营销领域。这使得生成个性化产品推荐和引人入胜的产品描述成为可能，帮助企业吸引并留住客户。

这些示例凸显了 LangChain 在创建解决方案方面的多功能性，这些方案能够满足从改善客户互动到支持医疗保健提供者、再到增强电商策略等广泛领域的复杂需求。

### LangChain 应用示例 – 文章摘要生成器

**工作流程**

- **安装必要包：** 首先安装所需的包：`requests`、`newspaper3k` 和 `langchain`。

- **数据收集：** 使用 `requests` 包从特定文章 URL 获取内容。

- **提取信息：** 利用 `newspaper` 包解析收集到的 HTML，提取文章标题和正文。

- **预处理文本：** 清理并结构化提取的内容，为输入到 ChatGPT 做准备。

- **生成摘要：** 使用 ChatGPT 生成文章内容的简洁摘要。

- **显示结果：** 将摘要与原始标题一同呈现，快速了解每篇文章的要点。

这个利用 ChatGPT 的应用，能让你借助 AI 摘要功能快速掌握文章的关键信息。它旨在让你无需花费大量时间阅读全文即可保持信息灵通，展示了 AI 在提升信息消费效率方面的实用性。

首先，获取你的 OpenAI API 密钥，这是使用摘要生成器的前提。这需要在 OpenAI 网站上创建一个账户并获取 API 访问权限。账户设置完成后，找到 API 密钥部分以获取你的密钥。

通过执行以下命令确保安装必要的库：`pip install langchain==0.1.4 deeplake openai==1.10.0 tiktoken`。同时，安装 `newspaper3k` 库也很重要，特别是版本 `0.2.8`，因为本教程已验证该版本的兼容性。

在你的 Python 脚本或笔记本中，将你的 API 密钥分配给名为 `OPENAI_API_KEY` 的环境变量。要使用 `.env` 文件实现此操作，请使用 `load_dotenv` 函数。

**在以下应用中：**

1.  你需要选择一篇文章的 URL 来创建摘要。将其添加到值为 `YOUR-URL` 的变量中。

2.  随后的脚本利用 `requests` 库和自定义的 `User-Agent` 头从一组 URL 中获取文章。

3.  接着，它使用 `newspaper` 库分离每篇文章的标题和内容。

4.  你需要使用命令 `pip install python-dotenv` 安装 Python Dotenv。

5.  要生成 `.env` 文件，请使用终端进入你的项目目录，并按如下方式执行 `touch` 命令：`touch .env`。

6.  在那里以 `variable = OPEN_AI_KEY = Your API Key` 的形式添加你的 API 密钥。

7.  然后加载环境变量：

    ```
    from dotenv import load_dotenv
    load_dotenv()
    ```

**然后使用以下代码：**

```
import json
from dotenv import load_dotenv
load_dotenv()
import requests
from newspaper import Article
headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}
article_url = "YOUR-URL"
session = requests.Session()
try:
response = session.get(article_url, headers=headers, timeout=10)
if response.status_code == 200:
article = Article(article_url)
article.download()
article.parse()
print(f"Title: {article.title}")
print(f"Text: {article.text}")
else:
print(f"Failed to fetch article at {article_url}")
except Exception as e:
print(f"Error occurred while fetching article at {article_url}: {e}")
```

**示例输出：**

```
Title: Meta claims its new AI supercomputer will set records
Text: Ryan is a senior editor at TechForge Media with over a decade of experience covering the latest technology and interviewing leading industry figures. He can often be sighted at tech conferences with a strong coffee in one hand and a laptop in the other. If it's geeky, he’s probably into it. Find him on Twitter (@Gadget_Ry) or Mastodon (@gadgetry@techhub.social)
Meta (formerly Facebook) has unveiled an AI supercomputer that it claims will be the world’s fastest.
```

## Hugging Face

虽然“Hugging Face”这个词可能会让许多人联想到一个友好的表情符号 ![](img/613844_1_En_7_Figa_HTML.gif)，但在技术社区中，它代表着更为重要的东西：一个类似于机器学习（ML）领域的“GitHub”的中心枢纽，致力于通过开源协作进行自然语言处理（NLP）和机器学习模型的协同开发、训练与部署。

Hugging Face 的突出特点在于它提供了**预训练模型**。这一关键创新意味着开发者不再需要从零开始启动项目；相反，他们可以利用这些现成的模型，根据自身特定需求进行调整，从而简化开发流程。

Hugging Face 是数据科学家、研究人员和机器学习工程师分享见解、寻求支持并为更广泛的开源运动做出贡献的重要聚集地。Hugging Face 将自己定位为“*构建未来的人工智能社区*”，其理念深深植根于社区驱动的进步。

该平台的快速扩张也归功于其用户友好的设计，无论是初学者还是经验丰富的专业人士都能轻松上手。通过努力积累最广泛的 NLP 和 ML 资源，Hugging Face 的使命是让 AI 技术大众化，使其广泛惠及全球用户。



### Hugging Face 的发展历程

Hugging Face 于 2016 年作为一家美法合资企业起步，最初专注于打造一款面向青少年的 AI 驱动聊天机器人。公司的转折点出现在它决定将聊天机器人的底层模型向全球开源，这一举措使其发展轨迹转向为 AI 领域提供强大且易于使用的工具。

2018 年发布的变革性 `Transformers` 库是 Hugging Face 发展史上的里程碑，它将 `BERT` 和 `GPT` 等预训练模型引入 AI 社区，这些模型迅速成为 NLP 任务的基础工具。

在随后的几年里，Hugging Face 深刻重塑了机器学习格局。其对开源协作的承诺激发了 NLP 领域的创新浪潮，培育了共同成长与技术进步的社区文化。

Hugging Face 已发展成为模型与数据集交换的核心枢纽，加速了 AI 领域的研究进展与实际应用。

### Hugging Face 的核心组件

Hugging Face 已成为自然语言处理（NLP）领域的基石，提供了一套满足多样化语言处理需求的工具和资源。以下是其核心组件与功能的概述。

#### Transformers 库

Hugging Face 的核心是 `Transformers` 库，这是一个专为 NLP 任务量身定制的尖端机器学习模型集合。该库包含大量预训练模型，适用于文本分析、内容生成、语言翻译和摘要创建等应用。`pipeline()` 方法的引入简化了将这些复杂模型应用于实际场景的过程，为各种 NLP 任务提供了直观的 API。该库对于普及先进 NLP 技术至关重要，使用户能够轻松定制和部署复杂的模型。

#### Hugging Face Hub

Hugging Face Hub 是一个动态的在线仓库，在撰写本书时，它拥有超过 35 万个模型、7.5 万个数据集和 15 万个演示应用（称为 Spaces）的惊人收藏，所有这些资源都是开源且免费访问的。该平台被设计为一个协作生态系统，使个人能够发现、实验、协作和开发机器学习技术。作为关键的聚集地，Hub 促进了机器学习项目的探索与创建，鼓励社区内的开放协作与创新。

Hugging Face Hub 采用基于 Git 的仓库来对所有相关文件进行版本控制管理。这包括以下内容：

- **模型：** 专为 NLP、视觉和音频处理任务量身定制的全面尖端模型集合
- **数据集：** 涵盖各种领域和类型的广泛数据集合，支持多样化的机器学习项目
- **Spaces：** 允许在网页浏览器中直接演示机器学习模型的交互式应用

此外，Hub 还配备了版本控制、提交历史、差异对比、分支以及与十多个库的集成等功能。如需深入了解这些功能，可查阅仓库文档。

#### 模型中心

作为社区的枢纽，模型中心（图 7-1）是用户探索和分享大量模型与数据集的地方。该功能促进了 NLP 开发的协作环境，使从业者能够贡献自己的模型，并从社区的集体智慧中受益。模型中心在 Hugging Face 网站上易于导航，提供多种筛选器帮助用户找到适合特定任务的模型。该中心对于培育一个动态、不断发展的生态系统至关重要，新模型在此定期添加和优化。

![图 7-1](img/613844_1_En_7_Fig1_HTML.jpg)

**图 7-1** 模型中心 - Hugging Face 可用模型

在超过 20 万个模型的庞大库中，你可以访问广泛的功能，包括以下内容：

- **自然语言处理（NLP）：** 涵盖语言翻译、内容摘要和文本生成等多种任务。这些能力构成了 OpenAI 的 GPT-3 通过 ChatGPT 提供的核心功能。
- **音频处理：** 你可以在此进行自动语音识别、检测说话（语音活动检测）或将文本转换为语音等操作。
- **计算机视觉：** 这些模型使计算机能够解释和理解来自世界的视觉信息。应用包括估计物体距离（深度估计）、图像分类以及图像到图像的转换。此类技术对于自动驾驶汽车等发展至关重要。
- **多模态模型：** 这些高级模型能够处理多种数据类型（文本、图像和音频）并生成输出。这种多功能性使其能够跨不同媒体实现广泛的应用。

#### 分词器

作为文本预处理的关键，分词器将语言分解为可管理的片段（即 token），机器学习模型随后使用这些 token 来理解和生成人类语言。这些 token 可以代表单词、子词或字符，对于将文本转换为机器可读格式至关重要。Hugging Face 的分词器针对与 `Transformers` 库的兼容性进行了优化，确保了对各种语言和文本格式的高效文本预处理。



