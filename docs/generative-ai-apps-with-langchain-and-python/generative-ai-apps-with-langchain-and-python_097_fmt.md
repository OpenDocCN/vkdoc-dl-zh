# 运行顺序链

`topic = "Artificial Intelligence"`

`result = sequential_chain({"topic": topic})`

`print(result["summary"])`

在此示例中，您有三个链：`topic_chain`、`data_chain` 和 `summary_chain`。我们使用 `SequentialChain` 类按顺序组合它们，并指定它们的执行顺序。`topic_chain` 的输出成为 `data_chain` 的输入，而 `data_chain` 的输出成为 `summary_chain` 的输入。

最后，您使用一个主题运行顺序链，并打印汇总结果。就是这么简单！

## SequentialChain 用例示例 1：客户支持聊天机器人应用

设想构建一个能够处理各种查询的客户支持聊天机器人。您可以组合链来创建一个强大且高效的聊天机器人工作流。

您可以使用 `RouterChain` 根据意图（例如，销售查询、技术支持、一般咨询）将用户查询路由到相应的链。

在每个特定意图的链中，使用 `SequentialChain` 将查询分解为更小的步骤，例如理解问题、检索相关信息以及生成响应。

如果需要，您可以在每个特定意图的链中使用 `ConditionalChain` 来处理特定情况或异常。

## SequentialChain 用例示例 2：内容生成管道应用

考虑一个内容生成管道，它接收一个主题并生成一篇结构良好的文章。您可以组合链来简化流程。

使用 `SequentialChain` 定义整体管道流程：

- 链 1：首先，根据主题生成文章大纲。

- 链 2：针对每个大纲要点，生成详细的段落。

- 链 3：然后，将生成的段落组合成一篇连贯的文章。

- 链 4：最后，校对并润色文章。

在顺序链的每个步骤中，您可以使用额外的链或工具来增强生成过程，例如从知识库中检索相关信息或应用语言模型进行文本润色。

## SequentialChain 用例示例 3：金融领域的自动欺诈检测

一家金融机构集成了 `SequentialChain` 来增强其欺诈检测系统，自动化分析交易模式以发现欺诈活动迹象。

**实施**：`SequentialChain` 分阶段处理交易，首先标记异常模式，然后与历史数据进行交叉引用，最后应用预测模型来评估欺诈概率。

**成果**：这种方法将欺诈交易的检测率提高了 25%，显著减少了损失并提高了客户账户的安全性。

这些只是几个示例，旨在让您了解链组合如何在现实场景中应用。

## 使用路由链的任务分配应用

当您希望根据预定义的规则或标准将输入路由到不同的链时，您将使用路由链。

以下是一个路由链工作流的示例：

```python
from langchain.chains import LLMChain, RouterChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# 链 1：处理销售查询
sales_prompt_template = "Respond to the following sales query: {query}"
sales_prompt = PromptTemplate(template=sales_prompt_template, input_variables=["query"])
sales_chain = LLMChain(llm=OpenAI(), prompt=sales_prompt)

# 链 2：处理支持查询
support_prompt_template = "Respond to the following support query: {query}"
support_prompt = PromptTemplate(template=support_prompt_template, input_variables=["query"])
support_chain = LLMChain(llm=OpenAI(), prompt=support_prompt)

# 根据主题将查询路由到相应链的路由链
router_chain = RouterChain.from_routes(
    routes=[
        ("sales", sales_chain),
        ("support", support_chain),
    ],
    default_chain=sales_chain,
    input_key="query",
    output_key="response",
)

# 运行路由链
query = "I have a question about my order"
result = router_chain.run(query)
print(result)
```

在此示例中，您有两个链，即 `sales_chain` 和 `support_chain`。您创建了一个 `RouterChain`，它根据主题将输入查询路由到相应的链。如果查询包含单词 "sales"，则将其定向到 `sales_chain`。如果包含单词 "support"，则将其定向到 `support_chain`。如果没有匹配的条件，则回退到默认的 `sales_chain`。最后，您使用一个查询运行路由链并打印响应。

## 案例研究：使用 LangChain 路由链增强客户服务

一家大型零售公司实施了 LangChain 路由链来自动化客户咨询路由，旨在提高响应速度和解决客户问题的准确性。

**实施与挑战**：路由链根据 "return"、"warranty" 和 "payment" 等关键词对咨询进行分类，并将其引导至相应的部门。挑战包括初始路由错误以及员工对自动化系统的抵触。解决方案包括根据服务代表的反馈优化路由逻辑，并提供员工培训以提高采用率。

**成果**：实施后，该公司每次咨询的处理时间减少了 30%，并且由于响应更快、更准确，客户满意度得到了提高。路由链的适应性通过一种学习机制得到增强，该机制随着时间的推移提高了其准确性。

**结论**：本案例研究展示了路由链在简化客户服务运营方面的有效性，证明了其在零售环境中显著提高运营效率和客户满意度的潜力。

## 使用条件链的情感分析应用

您将使用条件链根据特定条件执行不同的链。

以下是一个条件链工作流的示例：

```python
from langchain.chains import LLMChain, ConditionalChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# 链 1：积极情感
positive_prompt_template = "Respond positively to the following query: {query}"
positive_prompt = PromptTemplate(template=positive_prompt_template, input_variables=["query"])
positive_chain = LLMChain(llm=OpenAI(), prompt=positive_prompt)

# 链 2：消极情感
negative_prompt_template = "Respond cautiously to the following query: {query}"
negative_prompt = PromptTemplate(template=negative_prompt_template, input_variables=["query"])
negative_chain = LLMChain(llm=OpenAI(), prompt=negative_prompt)

# 基于情感分析的条件链
sentiment_conditions = [
    ("positive", positive_chain),
    ("negative", negative_chain),
]

conditional_chain = ConditionalChain.from_conditions(
    conditions=sentiment_conditions,
    default_chain=positive_chain,
    input_key="query",
    output_key="response",
)
```