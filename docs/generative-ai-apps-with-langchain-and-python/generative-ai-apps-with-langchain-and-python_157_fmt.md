# 基于决策执行操作

`input("按回车键获取下一批数据...")` 在此示例中，你拥有用于数据检索（`data_retrieval_tool`）和数据分析（`data_analysis_tool`）的虚拟函数。`data_retrieval_tool` 通过生成随机的温度和湿度值来模拟实时数据检索。`data_analysis_tool` 通过检查温度和湿度是否超过特定阈值来执行简单的分析。

该代理使用数据检索和分析工具以及决策提示进行初始化。`process_data_and_make_decision` 函数检索最新数据，对其进行分析，并利用代理根据洞察做出决策。

示例用法展示了一个持续循环，系统在其中检索数据、做出决策，并提示用户获取下一批数据。

这是一个用于说明概念的简化示例。在实际场景中，你需要将代理与实际的实时数据源集成，使用更复杂的数据分析技术，并根据决策实施必要的操作。

## 关键要点

在本章中，我们成功使用 LangChain 创建了一个自定义代理。

我们探讨了加载语言模型、定义工具和创建提示。此外，我们还介绍了代理的实际应用，例如客户支持自动化、个性化推荐和实时数据分析。通过添加记忆功能，我们增强了代理进行更自然、更连贯对话的能力。

## 复习题

让我们测试一下你对本章内容的理解。

1. 创建自定义代理的第一步是什么？

   A. 定义工具

   B. 加载语言模型

   C. 创建提示

   D. 将工具绑定到语言模型

2. 哪个函数允许代理记住之前的交互？

   A. `AgentAction`

   B. `ConversationBufferMemory`

   C. `ChatPromptTemplate`

   D. `AgentExecutor`

3. 代理在客户支持中的一个实际应用是什么？

   A. 生成随机数据

   B. 自动处理常见客户咨询

   C. 管理系统更新

   D. 分析服务器日志

4. 哪个工具用于检索最新一批实时数据？

   A. 数据分析工具

   B. 数据检索工具

   C. 实时数据工具

   D. 决策工具

5. 为代理添加记忆功能的好处是什么？

   A. 更快的响应时间

   B. 增强上下文感知的交互

   C. 提高计算准确性

   D. 简化代码库

## 答案

1. B

2. B

3. B

4. B

5. B

## 延伸阅读

以下参考资料将为你提供深入的知识和实际示例，帮助你有效理解和实现各种 LangChain 代理用例，从客户支持自动化到实时数据分析和个性化推荐：

1. **创建自定义代理**

   这是一本操作指南，展示了如何使用 LlamaIndex 构建自定义代理。

   [`docs.llamaindex.ai/en/latest/examples/agent/custom_agent/`](https://docs.llamaindex.ai/en/latest/examples/agent/custom_agent/)

2. **在代理中定义和使用工具**

   LangChain 工具集成：探索如何定义各种工具并将其集成到你的代理中，以增强其功能和有效性。

   [`python.langchain.com/v0.2/docs/integrations/platforms/`](https://python.langchain.com/v0.2/docs/integrations/platforms/)

3. **为代理设计提示**

   使用 LangChain 进行提示工程：创建有效提示的最佳实践和技术，以指导代理行为并优化响应。

   [`python.langchain.com/v0.1/docs/modules/model_io/prompts/`](https://python.langchain.com/v0.1/docs/modules/model_io/prompts/)

4. **实际用例和示例**

   LangChain 用例：LangChain 代理在各种应用中的实际用例和示例集合，为你的项目提供见解和灵感。

   [`js.langchain.com/v0.1/docs/use_cases/`](https://js.langchain.com/v0.1/docs/use_cases/)