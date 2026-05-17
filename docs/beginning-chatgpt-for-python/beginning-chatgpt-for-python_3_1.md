# 3. 在 Python 中创建一个基本的 ChatGPT 客户端

本章的目的简单明了。我们将仅用几行 Python 代码构建最强大的 ChatGPT 客户端。这个客户端将比你在 ChatGPT 网站上能做的多得多，并且会为你提供比我们在第 1 章中看到的 Chat Playground 更多的选项。

## 创建我们的 ChatGPT 聊天补全应用程序 `chatgpt_client.py`

清单 3-1 是我们 ChatGPT 客户端 `chatgpt_client.py` 的代码。

```
"""
演示使用 OpenAI 的 GPT-4 进行聊天补全的 API 调用的脚本。
"""
from dotenv import load_dotenv
from openai import OpenAI
### 从 .env 加载环境变量
load_dotenv()
### 实例化 OpenAI 对象
client = OpenAI()
response = client.chat.completions.create(
model="gpt-4o",
messages=[
{
"role": "system",
"content": "你是一名 Python 开发者"
},
{
"role": "user",
"content": "为什么 Python 通常用于数据科学？"
}
],
temperature=0.85,
max_tokens=1921,
top_p=1,
frequency_penalty=0,
presence_penalty=0
)
print(response)
清单 3-1
chatgpt_client.py
```

当你分析清单 3-1 中的代码时，你会看到一些在 Chat Playground 中非常熟悉的东西，比如 `model`、`messages`、`temperature` 和 `tokens`。

注意

在本章中，我们将使用 Python 数据类型，因此你会看到一个 `list`，而在官方 OpenAPI 文档中指定的是 `Array`。

## 使用 `OpenAI.chat.completions.create()` 向 ChatGPT 发送消息

`OpenAI.chat.completions.create()` 方法基本上是你可以在 Chat Playground 中执行的操作的一对一映射；因此，这个方法应该让你感觉非常熟悉。

表 3-1 描述了调用 `OpenAI.chat.completions.create()` 方法所需参数的格式。虽然表格很长，但快速浏览后，你应该会看到只有少数几个字段是成功调用该方法所必需的。

该方法的响应被称为 `ChatCompletion`。

### 检查方法参数

表 3-1

创建 ChatCompletion 对象的结构



| 字段 | 类型 | 是否必填 | 描述 |
| --- | --- | --- | --- |
| `model` | `String` | `Required` | 用于`ChatCompletion`的模型 ID。兼容模型包括：`gpt-4`、`gpt-4-0613`、`gpt-4-32k`、`gpt-4-32k-0613`、`gpt-4o`、`gpt-4o-mini`、`o1`、`o1-mini` |
| `messages` | `List` | `Required` | 共有四种消息类型，每种类型各有其要求：`系统消息`（见表 3-2）、`用户消息`（见表 3-3）、`助手消息`（见表 3-4）、`工具消息`（见表 3-5） |
| `frequency_penalty` | `Number` 或 `null`（默认值：`0`） | `Optional` | 介于-2.0 到 2.0 之间的数字。正值会根据对话历史中已有的词频对词元进行惩罚，从而降低逐字重复相同句子的可能性。 |
| `logit_bias` | `JSON map`（默认值：`null`） | `Optional` | 允许你修改特定词元在补全中出现的可能性。你需要提供一个 JSON 对象，将词元（通过分词器中的词元 ID 指定）映射到-100 到 100 之间的相关偏差值。该偏差会在采样前添加到模型的 logits 中。 |
| `logprobs` | `Boolean` 或 `null` | `Optional`，默认为`false` | 此参数用于决定是否返回输出词元的对数概率。当设置为`true`时，它会提供消息内容中包含的每个输出词元的对数概率。不过，`gpt-4-vision-preview`模型目前不支持此功能。 |
| `max_tokens` | `Integer` 或 `null` | `Optional` | 此参数用于设置生成的`ChatCompletion`可以拥有的最大词元数量。 |
| `n` | `Integer` 或 `null`（默认值：`1`） | `Optional` | 指定模型应为每条输入消息生成多少个`ChatCompletion`选项。 |
| `presence_penalty` | `Number` 或 `null`（默认值：`0`） | `Optional` | 介于-2.0 到 2.0 之间的数字。正值会根据新词元是否出现在对话历史中进行惩罚，从而鼓励模型谈论新话题。 |
| `response_format` | `JSON object` | `Optional` | 你有两个选项：`{ "type": "json_object" }`用于 JSON 对象响应，或`{ "type": "text" }`用于文本响应。注意：务必记住，在 JSON 模式下操作时，你需要通过系统或用户指令明确命令模型生成 JSON。否则，模型可能会无限输出空白字符，直到达到词元上限，导致请求看似卡住。此外，请注意，如果`finish_reason`为`"length"`，则表明生成内容超出了`max_tokens`或对话超出了最大允许上下文长度，这可能导致消息被截断。 |
| `seed` | `Integer` 或 `null` | `Optional` | 通过指定种子，系统将尝试生成可重复的结果。理论上，这意味着如果你使用相同的种子和参数重复请求，应该会得到相同的结果。为了获取用于后续请求的种子值，请从你上一次的响应中复制`system_fingerprint`。 |
| `stop` | `String`/`list`/`null`（默认值：`null`） | `Optional` | 你可以提供最多四个序列，API 将在这些序列处停止生成更多词元。这对于控制响应的长度或内容很有用。 |
| `stream` | `Boolean` 或 `null`（默认值：`false`） | `Optional` | 如果`"stream"`设置为`"true"`，部分消息更新将以服务器发送事件的形式发送。这意味着词元将在可用时作为纯数据事件发送，并且流将以`"data: [DONE]"`消息结束。 |
| `temperature` | `Number` 或 `null`（默认值：`1`） | `Optional` | 有效值范围在 0 到 2 之间。控制模型输出的随机性。最佳实践是调整`top_p`或`temperature`，但不要同时调整两者。 |
| `tool_choice` | `String` 或 `JSON object` | `Optional` | 此参数控制模型调用哪个（如果有）函数。你有两个选项：`"none"`或`"auto"`。如果你不希望模型调用函数，请使用`"none"`。如果你希望模型在生成消息或调用函数之间进行选择，请使用`"auto"`。通过`{"type": "function", "function": {"name": "my_function"}}`指定特定函数会强制模型调用该函数。请注意，当没有函数时，默认值为`"none"`；当存在函数时，默认值为`"auto"`。 |
| `tools` | `List` | `Optional` | 你可以选择指定模型可能调用的工具列表。目前，仅支持函数作为工具。使用此参数提供模型可能为其生成 JSON 输入的函数列表。 |
| `top_logprobs` | `Integer` 或 `null` | `Optional` | 可以是 0 到 5 之间的任意整数。用于确定在每个词元位置返回的最可能词元的数量，并附带其各自的对数概率。要使此参数生效，必须通过将`logprobs`设置为`true`来启用它。 |
| `top_p` | `Number` 或 `null`（默认值：`1`） | `Optional` | 有效值范围在 0 到 1 之间。指示是考虑少数可能性（0）还是所有可能性（1）。最佳实践是调整`top_p`或`temperature`，但不要同时调整两者。 |
| `user` | `String` | `Optional` | 这是一个唯一 ID，你可以选择生成它来表示你的最终用户。这将有助于 OpenAI 监控和检测滥用行为。 |

好了，表 3-1 看起来有点吓人！不过，如前所述，只有`model`和`messages`是必填参数。

此外，我们还在上面的代码清单 3-1 中展示了这些参数在实际应用程序中是如何使用的。

所以，正如你所见，作为一名 Python 开发者，我们有几个可供使用的选项和参数，这些是普通人使用 ChatGPT 网站或聊天游乐场无法做到的。

现在，最需要详细解释的参数是`messages`参数，让我们进一步分析它。

## 共有四种消息类型

当以编程方式调用 ChatGPT API 时，你可以向 API 提供四种类型的消息：

*   系统消息
*   用户消息
*   助手消息
*   工具消息

好消息是，如果你回顾一下第 1 章中我们解释如何使用聊天游乐场的内容，你会发现我们已经接触过前三种消息类型了！我们目前不熟悉的唯一新消息类型是工具消息。

### 系统消息（字典）

表 3-2：系统消息的结构

| 字段 | 类型 | 是否必填 | 描述 |
| --- | --- | --- | --- |
| `role` | `String` | `Required` | 必须设置为字符串`"system"` |
| `content` | `String` | `Required` | 这些是你希望系统在对话中执行的指令。 |
| `name` | `String` | `Optional` | 这是你可以为系统提供的可选名称。 |

代码清单 3-2 是代码清单 3-1 中的一个片段，展示了系统消息的格式。

```
messages=[
{
"role": "system",
"content": "你是一名 Python 开发者"
},
...
```


### 用户消息（字典）

表 3-3  
用户消息的结构

| 字段 | 类型 | 是否必填 | 描述 |
| --- | --- | --- | --- |
| `role` | `String` | `Required` | `必须设置为字符串 "user"` |
| `content` | `String` | `Required` | `此字符串包含您想要发送给 ChatGPT 的实际消息或问题。` |
| `name` | `String` | `Optional` | `这是您在对话中可选的名称。` |

清单 3-3 是清单 3-1 中的一个片段，展示了用户消息的格式。

```
messages=
...
{
"role": "user",
"content": "为什么 Python 通常用于数据科学？"
}
...
清单 3-3
格式化用户消息
```

### 助手消息（字典）

**注意**  
以防您忘记，助手消息用于“提醒”ChatGPT 它在之前回复中告诉您的内容。理想情况下，这可以让您继续数周或数月前与它的对话。

表 3-4  
助手消息的结构

| 字段 | 类型 | 是否必填 | 描述 |
| --- | --- | --- | --- |
| `role` | `String` | `Required` | `必须设置为字符串 "assistant"` |
| `content` | `String` | `Required` | `此字符串包含来自之前对话中 ChatGPT 的回复。` |
| `name` | `String` | `Optional` | `这是您在对话中为 ChatGPT 提供的可选名称。` |
| `tool_calls` | `List` | `Optional` | `如果 ChatGPT 在之前的回复中使用了工具，请在此处包含它指定的工具。` |
| `↳ id` | `String` | `Required` | `这是 ChatGPT 调用的工具的 ID。` |
| `↳ type` | `String` | `Required` | `这是 ChatGPT 调用的工具的类型。只有字面量 "function" 是有效的工具。` |
| `↳ function` | `Object` | `Required` | `这是模型调用的函数。` |

清单 [3-4 是清单 3-1 中的一个片段，展示了助手消息的格式。

```
messages=
...
{
"role": "assistant",
"content": "Python 通常用于数据科学有几个原因..."
}
...
清单 3-4
格式化助手消息
```

### 工具消息（字典）

工具消息是一种高级消息类型，用于非常特定的用例。您不能在 ChatGPT 网站或聊天游乐场中使用它们。通过使用工具消息和表 [3-1 中的 `tool` 参数，您可以让 ChatGPT 为您“调用一个函数”。

乍一看，您可能会想：“哇！ChatGPT 会在云端加载我的代码并为我执行它？太棒了！”不幸的是，事实并非如此。

通过提供函数名称和调用它所需的参数，ChatGPT 会告诉您是否应该调用该函数以及要放入函数的参数。然后，您需要在您的 Python 代码中*自己*调用该函数。

表 3-5  
工具消息的结构

| 字段 | 类型 | 是否必填 | 描述 |
| --- | --- | --- | --- |
| `role` | `String` | `Required` | `必须设置为字符串 "tool"` |
| `content` | `String` | `Required` | `此字符串包含工具消息的内容。` |
| `tool_call_id` | `List` | `Optional` | `这是工具调用的 ID。` |

## 运行 `chatgpt_client.py`

那么，在运行我们在清单 3-1 中创建的代码后，我们可以预期得到一个类似清单 3-5 所示的响应。

```
ChatCompletion(id='chatcmpl-9ACnRg1bk54jYeIFbxJ3yDnomQmij', choices=[Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content="Python 通常用于数据科学有几个原因：\n\n1\. **简单易读**：Python 的语法清晰直接，使代码易于阅读和编写。这种简单性使数据科学家能够快速理解 Python 语法并开始编码。\n\n2\. **丰富的库生态系统**：Python 拥有大量专为数据科学任务设计的库和工具。例如 NumPy 和 SciPy 适用于科学计算，Pandas 擅长数据处理和分析，Matplotlib 和 Seaborn 用于数据可视化，Scikit-learn 用于机器学习。\n\n3\. **支持多种数据格式**：Python 支持数据科学中常用的多种数据格式。您可以轻松加载和处理 CSV、Excel 电子表格、SQL 数据库等不同格式的数据。\n\n4\. **社区支持**：Python 拥有庞大且活跃的开发者社区，他们不断为改进语言及其工具做出贡献。这也意味着当问题出现时，通常比使用不太流行的语言更容易找到解决方案和示例。\n\n5\. **集成能力**：Python 可以轻松与 C、C++、Java 等其他语言集成，并且几乎可以在所有操作系统上运行。这使得它成为需要处理不同软件和系统的数据科学家的便捷选择。\n\n6\. **支持高级数据分析**：Python 支持各种类型的高级数据分析，包括机器学习、人工智能和深度学习，拥有 TensorFlow、PyTorch 和 Keras 等库。\n\n7\. **适合原型开发**：Python 的简单性和速度使其非常适合原型开发。数据科学家可以使用 Python 构建模型，查看其工作方式，然后在必要时使用 Python 或其他语言构建更永久的版本。", role='assistant', function_call=None, tool_calls=None), logprobs=None)], created=1712219497, model='gpt-4-0613', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=368, prompt_tokens=25, total_tokens=393))
清单 3-5
ChatGPT 解释为什么 Python 如此强大
```

因此，当我们简要查看清单 3-5 时，我们看到响应（称为 `ChatCompletion`）的主体部分是对我们在本章开头清单 3-1 中向 ChatGPT 提出的问题的回答。然而，我们的响应周围有很多元数据，让我们更详细地分析一下 `ChatCompletion` 对象。



### 处理响应（`ChatCompletion`）

**表 3-6** `ChatCompletion` 对象响应的结构

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| `id` | `String` | `ChatCompletion` 的唯一标识符。 |
| `choices` | `List` | `ChatCompletion` 选项的列表。如果表 3-1 中的 `n` 大于 1，则响应中可能存在多个选项。 |
| `↳ finish_reason` | `String` | 每个响应都会包含一个 `finish_reason`。`finish_reason` 的可能值为：`stop`、`length`、`tool_call`、`content_filter`、`null`。 |
| `↳ index` | `Integer` | 该选项在选项列表中的索引。 |
| `↳ message` | `Object` | 模型生成的 `ChatCompletionMessage`。详细信息见表 3-7。 |
| `↳ logprobs` | `Object` 或 `null` | 该选项的对数概率信息。 |
| `model` | `String` | 用于 `ChatCompletion` 的模型。 |
| `system_fingerprint` | `String` | 如果你希望从之前的对话中获得可重现的结果，可以在后续请求中将此参数用作 `seed`。 |
| `object` | `String` | 此字段始终返回字面量 `"chat.completion"`。 |
| `usage` | `Object` | 补全请求的使用统计信息。 |
| `↳ completion_tokens` | `Integer` | 生成的补全内容中的令牌数。 |
| `↳ prompt_tokens` | `Integer` | 提示中的令牌数。 |
| `↳ total_tokens` | `Integer` | 请求中使用的令牌总数，包括提示和补全内容。 |

`ChatCompletion` 对象中最重要的部分是 `ChatCompletionMessage`，其详细信息见表 3-7。

### `ChatCompletionMessage`

**表 3-7** `ChatCompletionMessage` 的结构

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| `role` | `String` | 此字段始终为字面量 `"assistant"`。 |
| `content` | `String` 或 `null` | 这是一个字符串，包含 ChatGPT 对我们请求的响应。 |
| `tool_calls` | `List` | 如果你在表 3-1 中指示 ChatGPT 调用一个工具（目前是一个函数），那么此列表将存在于 `ChatCompletionMessage` 中。 |
| `↳ id` | `String` | 这是 ChatGPT 调用的工具的 ID。 |
| `↳ type` | `String` | 这是 ChatGPT 调用的工具的类型。只有字面量 `"function"` 是有效的工具。 |
| `↳ function` | `Object` | 这是模型调用的函数及其参数。 |

## 结论

在本章中，我们借鉴了第 1 章和第 2 章的经验，用 Python 创建了一个功能完备的 ChatGPT 客户端。在 ChatGPT 客户端的代码中，我们看到了一些在 Chat Playground 中已经介绍过的术语，例如 `model`、`messages`、`temperature` 和 `tokens`。

我们还看到，作为 Python 开发者，OpenAI 为我们提供了大量额外的选项来调用 ChatGPT，这些选项是普通日常用户甚至使用 Chat Playground 的技术人员都无法使用的。在本章中，我们花时间解释了这些选项，重点是我们能够发送的 `messages`。

既然我们已经有了一个可用的 Python ChatGPT 客户端，接下来让我们看看如何利用它来完成本书其余部分的示例！

