# 排版后的内容

遇到决策点时，你需要根据链已处理的数据做出选择。这些决策点将充当网关，根据预定义条件引导链的流程。你将使用条件逻辑或路由器链来实现这些决策点。

4. **终点**：这是链结束执行并返回最终输出的位置。终点代表链任务的完成，无论是生成响应、更新数据库还是触发另一个操作。

以下是一个使用`LangChain`实现的客户支持聊天机器人示例：

- **触发器**：用户向聊天机器人发送消息。

- **步骤**：
  - 步骤 1：你处理用户的消息并将其作为输入传递给`LLM`。
  - 步骤 2：`LLM`根据输入和预定义提示生成响应。
  - 步骤 3：然后你使用工具（例如情感分析、实体识别）处理响应。

第 6 章 使用链构建智能聊天机器人和自动化分析系统

- **决策点**：
  - 如果用户的消息被归类为投诉，你可以将对话转接给人工客服。
  - 如果用户的消息是常见问题，你可以提供预定义答案。

- **终点**：然后你通过将最终响应发送回用户来完成对话流程。

### 步骤中的内部组件

在此示例中，内部组件（`LLM`、提示、工具）在每个步骤中协同工作，而高级组件（触发器、步骤、决策点、终点）定义了链的整体结构和流程。

语言模型（`LLM`）是每个链的核心，负责理解和生成类似人类的文本。提示通过提供上下文和指令来引导`LLM`生成相关且连贯的响应。

工具允许链与外部世界交互，例如从`API`获取数据、查询数据库或执行特定任务。某些链还具有内存组件，用于在多次交互中保留上下文。

## 链的类型

现在你已经了解了基本组件，让我们探索在`LangChain`中你将看到的两种主要链类型：`LCEL`（`LangChain`执行语言）链和传统链。将`LCEL`链视为更现代、更灵活的链，而传统链则更直接，适用于较简单的任务。

第 6 章 使用链构建智能聊天机器人和自动化分析系统

### `LCEL`链

`LCEL`代表“`LangChain`执行语言”，是创建链的现代方式。`LCEL`允许你使用专门为此目的设计的领域特定语言，对每个步骤进行精细控制来定义链：

- **灵活性**：你可以定义复杂逻辑并无缝集成各种操作。
- **可扩展性**：由于其模块化特性，你可以轻松扩展和修改它们，使其非常适合不断增长的应用程序。
- **用例**：假设你需要开发一个系统，该系统获取用户数据、分析数据，然后动态生成个性化报告。你可以使用`LCEL`链精确构建此工作流，并有效处理任务的每个方面。

#### 传统链

**传统链**是在`LangChain`中构建链的原始方法。虽然它们不如`LCEL`链灵活，但你会发现它们更易于使用，因为它们已预先构建好以处理特定任务。

- **简单性**：与`LCEL`链相比，这些链实现起来更直接，所需的设置和配置更少。
- **直接应用**：你可以为工作流稳定且较少更改或定制的应用程序选择它们。

第 6 章 使用链构建智能聊天机器人和自动化分析系统

- **用例**：如果你的应用程序需要执行标准任务，例如根据用户查询发送格式化电子邮件响应，那么传统链可能是完美的选择。

### `LCEL`与传统链的区别

最好看几个创建`LCEL`和传统链的示例，以理解它们之间的区别。

#### `LCEL`链示例

以下是一个`LCEL`链的简单示例：

```python
# LCEL Chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import load_chain

template = "What is the capital of {country}?"
prompt = PromptTemplate(template=template, input_variables=["country"])
chain = load_chain("llm_chain", llm=OpenAI(), prompt=prompt)
result = chain.run("France")
```

#### 传统链示例

以下是一个传统链的简单示例：

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

template = "What is the capital of {country}?"
prompt = PromptTemplate(template=template, input_variables=["country"])
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("France")
```

如果你的代码因版本问题无法运行，也可以尝试 `chain = prompt | llm`。使用 `chain.invoke("France")` 即可生效。

如你所见，传统链是通过 `LLMChain` 类构建的，而 LCEL 链则使用 `load_chain` 函数加载。LCEL 链提供了更简洁、更灵活的方式。该链由一个 OpenAI 大语言模型和一个提示模板组成。LCEL 链支持流式传输、异步执行和自动可观测性等高级功能，使其成为大多数现代应用的首选。

那么，你应该选择哪一种呢？接下来我们讨论一下。

### 何时使用不同类型的链

我们来讨论每种链适用的常见场景。

#### LCEL 链

以下是 LCEL 更适用的几种情况：

1. 当你需要定义复杂逻辑，并对步骤间的数据流有更高控制要求时，请使用 LCEL。
2. 如果你处理的是大型数据集，或需要实时处理数据，请使用 LCEL，因为它支持流式传输、异步执行和并行化。
3. 当你需要集成外部 API、数据库或服务时，请使用 LCEL 链来包含自定义工具和插件。

#### 传统链

在以下情况下使用传统链：

- 当你的用例简单直接，不需要复杂逻辑或自定义集成时，请使用传统链。它们非常适合快速原型设计和实验。
- 如果你处理的是定义明确且稳定的数据集。
- 当你刚开始接触 LangChain，想初步了解链的工作原理时，传统链是一个很好的起点。

## 使用 LCEL 链进行构建

既然我们已经讨论了何时使用 LCEL 链，接下来就讨论如何创建它们。在构建 LCEL 链时，主要需要考虑两个方面，即构建链本身，然后根据需求对其进行定制。

![](img/index-240_1.jpg)

### 构建 LCEL 链

要构建一个 LCEL 链，你通常会使用 `load_chain` 函数，该函数负责用必要的组件（如语言模型、提示和工具）来初始化链。以下是一个简单示例：示例中提供的 `"llm_chain"` 模板可能会引起混淆，因为它可能并不直接存在于用户安装的 LangChain 中。这可能会让人误以为该模板是预定义的，但实际上，`load_chain` 函数可能需要自定义的链定义，而不是像 `"llm_chain"` 这样的预构建链。让我们调整一下解释，使其更清晰并回应这一评论。

**更新后的解释：**

要在 LangChain 中构建一个链，你通常会使用 `load_chain` 函数，该函数通过连接语言模型、提示和工具等必要组件来初始化链。然而，具体的链模板（例如本例中的 `"llm_chain"`）需要事先创建，或者从你在项目中定义的自定义设置中加载。LangChain 本身并不提供像 `"llm_chain"` 这样的预定义链。

以下是如何在不依赖 `load_chain` 的情况下手动创建一个简单的自定义链：

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 步骤 1：初始化你的语言模型
llm = OpenAI(temperature=0.9)

# 步骤 2：定义一个提示模板
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
```