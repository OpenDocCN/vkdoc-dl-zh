# 第 9 章：构建不同类型的代理

## 复习题

让我们测试一下您对本节内容的理解。

1. 设计代理的第一步是什么？
   - A. 实现代理的记忆
   - B. 定义代理的目标
   - C. 加载语言模型
   - D. 创建自定义工具

2. 以下哪项是 LangChain 中代理的核心概念？
   - A. `AgentHandler`
   - B. `AgentAction`
   - C. `AgentLoader`
   - D. `AgentManager`

3. 哪种类型的代理是为对话式 AI 应用设计的？
   - A. 零样本反应代理
   - B. 结构化工具代理
   - C. ReAct 代理
   - D. 对话代理

4. 在 LangChain v0.2 中，使用哪个函数来初始化代理？
   - A. `load_agent`
   - B. `start_agent`
   - C. `initialize_agent`
   - D. `create_agent`

5. 为代理添加记忆的好处是什么？
   - A. 减少代理的响应时间。
   - B. 允许代理记住之前的交互并提供上下文感知的响应。
   - C. 使代理能够执行并行处理。
   - D. 简化代理的代码库。

6. 哪个工具允许代理搜索网络信息？
   - A. `LLMTool`
   - B. `SerpAPI`
   - C. `OpenAITool`
   - D. `WikiQuery`

7. `AgentExecutor` 在 LangChain 中的作用是什么？
   - A. 初始化代理。
   - B. 处理工具加载。
   - C. 运行代理并执行其选择的动作。
   - D. 管理代理的记忆。

8. 哪种代理类型针对处理具有多个输入的工具进行了优化？
   - A. 零样本反应代理
   - B. ReAct 代理
   - C. 结构化聊天代理
   - D. 自问自答代理

9. 在为代理描述工具时，一个重要的考虑因素是什么？
   - A. 使工具名称尽可能短
   - B. 提供清晰简洁的描述
   - C. 使用复杂的 JSON 模式
   - D. 限制可用工具的数量

10. 使用 LangGraph 构建代理的主要优势是什么？
    - A. 简化了集成 API 的过程
    - B. 允许代理使用图结构处理复杂工作流
    - C. 减少了对记忆集成的需求
    - D. 提高了语言模型处理的速度

## 答案

1. B
2. B
3. D
4. C
5. B
6. B
7. C
8. C
9. B
10. B

## 延伸阅读

以下参考资料将为您提供深入的知识和实际示例，帮助您有效地理解和实现各种类型的 LangChain 代理：

1. **工具调用代理**：关于如何设置和使用工具调用代理动态处理各种任务的信息。
   - [`python.langchain.com/v0.1/docs/modules/agents/agent_types/tool_calling/`](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/tool_calling/)

2. **OpenAI 工具代理**：关于利用 OpenAI 工具代理实现各种功能并将其集成到项目中的详细文档。
   - [`python.langchain.com/v0.1/docs/modules/agents/agent_types/openai_tools/`](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/openai_tools/)

3. **结构化聊天代理**：关于实现结构化聊天代理以处理多输入工具和管理复杂交互的分步指南。
   - [`python.langchain.com/v0.1/docs/modules/agents/agent_types/structured_chat/`](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/structured_chat/)

4. **ReAct 代理**：关于设置和使用 ReAct 代理以实现推理和行动能力的详细说明和示例。

5. **LangGraph 增强能力**：了解如何使用 LangGraph 通过图结构创建更复杂、能力更强的代理。
   - [`blog.langchain.dev/langgraph/`](https://blog.langchain.dev/langgraph/)

6. **代理类型概览**：LangChain 中可用的不同类型代理及其特定用例的概述。
   - [`python.langchain.com/v0.1/docs/modules/agents/agent_types/`](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/)

