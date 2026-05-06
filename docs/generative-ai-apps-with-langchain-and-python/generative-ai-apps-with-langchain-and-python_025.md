# 第 1 章 LangChain 与大语言模型入门

- 探究直接调用 LLM API 所面临的挑战
- 通过实际对比、案例研究和练习，展示 LangChain 的优势
- 理解 LangChain 的架构，实现无缝集成、可扩展性与创新

## 用 LangChain 将你的创意变为现实

现在你已经对 LangChain 和 LLM 有了更深入的了解，知道它们如何协同构建出强大且令人惊叹的生成式 AI 应用。我想问你的问题是：你该如何将自己的创意变为现实？以下是一些建议：

**继续学习**：本章只是一个开始。在接下来的几章中，我们将深入探讨 LangChain 的每个组件，探索高级功能，并不断尝试新的项目创意。

**今天就迈出第一步**：访问 LangChain 的 GitHub 仓库，获取代码示例、最新更新和灵感。

[`github.com/langchain-ai/langchain`](https://github.com/langchain-ai/langchain)

**加入 LangChain 社区**：加入 LangChain 社区论坛，或在社交媒体上关注他们，与其他 LangChain 爱好者建立联系。分享你的想法，提出问题，并发现新的途径。

[`discord.com/invite/6adMQxSpJS`](https://discord.com/invite/6adMQxSpJS)

## 技术术语表

以下是本章中使用的技术术语定义列表：

**LangChain**：一个用于开发由语言模型驱动的应用程序的框架。它通过将语言模型连接到各种上下文来源，支持创建具有上下文感知和推理能力的应用。

**大语言模型（LLMs）**：能够理解、生成和与人类语言交互的先进 AI 系统。它们在海量数据集上训练，以执行各种与语言相关的任务。

**GPT-4**：由 OpenAI 开发的生成式预训练 Transformer 的一个版本，以其能够根据给定提示生成连贯且上下文相关的文本而闻名。

**PaLM**：由 Google 开发的语言模型，擅长理解和生成语言，以及解决跨多个领域的复杂问题。

**Gemini**：一个多模态 LLM 模型的示例，能够处理和理解多种数据类型，包括文本、图像和视频，适用于广泛的应用场景。

**提示工程**：精心设计输入提示以引导语言模型生成特定或期望输出的实践。它涉及构建提示结构，为准确有效的响应提供必要的上下文。

**数据感知**：应用程序与各种来源的数据进行交互并整合的能力，确保响应或操作基于相关且最新的信息。

**智能代理**：指能够自主执行操作或任务的应用程序，通常模仿人类行为，例如搜索网络、与数据库交互或填写表单。

**检索增强生成（RAG）**：一种在提示过程中向语言模型引入新信息的技术，以减少不准确性并提高响应质量。

**分词**：将文本分解为更小单元（词元）的过程，例如单词或短语，以便语言模型更容易处理。

## 延伸阅读

以下资源列表可帮助你巩固对本章所涵盖主题的理解，并进一步探索 LLM 和 LangChain 的世界：

**《更好的语言模型及其影响》（OpenAI 博客）**：虽然本文主要关注 GPT-2，但它提供了关于 LLM 如何理解和生成文本的宝贵见解，为提示工程奠定了基础。 [`openai.com/research/better-language-models`](https://openai.com/research/better-language-models)

**《面向知识密集型 NLP 任务的检索增强生成》（Google）**



### 研究博客

通过详细示例和在知识密集型任务中的应用，探讨了检索增强生成（RAG）的概念。[`arxiv.org/abs/2005.11401`](https://arxiv.org/abs/2005.11401)

### OpenAI API 文档

OpenAI API 的官方文档，涵盖了从身份验证到请求参数的方方面面。对于任何使用大语言模型构建应用程序的开发者来说，这都是必读资料。[`platform.openai.com/docs/introduction`](https://platform.openai.com/docs/introduction)

### Google Gemini 大语言模型文档

本文档包含一些笔记本、教程和其他示例，可帮助你快速上手 Gemini 模型。[`cloud.google.com/vertex-ai/generative-ai/docs/learn-resources`](https://cloud.google.com/vertex-ai/generative-ai/docs/learn-resources)

