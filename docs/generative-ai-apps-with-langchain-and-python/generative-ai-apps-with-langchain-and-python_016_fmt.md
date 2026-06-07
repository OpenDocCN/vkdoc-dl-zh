# 备注

LangChain 的数据连接组件不仅减少了开发时间，还能确保你的应用程序能够以最小的调整适应新的数据源或数据结构的变化。

## 第 1 章 LangChain 与 LLM 简介

### 构建上下文感知型应用

上下文感知型应用是指那些能够解读其运行环境中的上下文，并据此做出响应的应用。LangChain 通过利用其运行环境中的数据或用户交互来实现这一目标，从而提供个性化的用户体验。

#### 为何这很重要？

这很重要，因为它能让你的应用程序适应不断变化的环境条件或用户偏好，从而提供高度个性化、更具相关性和有效性的响应。

#### 示例案例研究：AI 辅导系统

开发一个辅导系统，通过根据学生的交互历史和其他上下文因素调整数学题的难度来帮助学生。

首先，你需要构建上下文感知能力。当学生与系统交互时，应用程序必须捕捉学生感到困难的主题，将学生归入特定的难度类别，然后调整未来向学生呈现问题的难度级别。它还可能利用一天中的时间、学生回答的准确率或回答速度来调整其教学风格。

接着，LangChain 通过使用提示模板，将难度级别和一天中的时间作为变量传递给 LLM，从而轻松地为学生提供个性化问题。当你学习第 5 章的提示模板后，你将能更好地理解这一点。

### 开发基于 RAG 的应用

借助 LangChain，你可以通过一项称为检索增强生成（RAG）的功能来开发高级应用。这一创新功能允许你将新的、相关的信息引入 LLM 的响应过程。

#### RAG 的工作原理

使用 RAG 就像让 LLM 查阅一个庞大的图书馆来提供最准确的答案。这一功能对于减少错误（通常称为“幻觉”）并提高应用程序所提供数据的准确性至关重要。

### 示例代码

以下是一个示例代码来说明这一点。别担心，我们将在第 5 章中进一步讨论，但这只是为了让你提前感受一下。

```python
#### 获取学生的交互历史
difficulty_level = get_student_difficulty_level(student_id)
time_of_day = get_current_time_of_day()
accuracy = get_student_accuracy(student_id)

#### 根据上下文生成个性化问题
prompt_template = PromptTemplate(
    input_variables=["difficulty_level", "time_of_day",
                     "accuracy"],
    template="生成一个{难度级别}的数学问题，适合在{一天中的时间}学习，并考虑学生{准确率}的准确率。"
)
personalized_question = chain.run(prompt_template,
                                  difficulty_level=difficulty_level, time_of_day=time_of_day,
                                  accuracy=accuracy)
```