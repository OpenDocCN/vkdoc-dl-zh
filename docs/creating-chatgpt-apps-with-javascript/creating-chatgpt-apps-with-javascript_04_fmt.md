# 3. 在 JavaScript 中创建基础 ChatGPT 客户端

本章的目的简单明了。我们将仅用几行 JavaScript 代码构建最强大的 ChatGPT 客户端。这个客户端将比你在 ChatGPT 网站上能做的多得多，并且将提供比我们在第 1 章中看到的 Chat Playground 更多的选项。

## 在 JavaScript 中创建我们的 ChatGPT 客户端应用程序

清单 3-1 是我们的 ChatGPT 客户端的 JavaScript 代码。这是一个简单的客户端，允许我们使用 JSON 创建 `system` 和 `user` 消息。我们还可以指定所需的模型和配置参数，例如要使用的令牌数量。

```javascript
import OpenAI from "openai";
import "dotenv/config";
// 创建一个新的 open ai 客户端
const openai = new OpenAI({
apiKey: process.env["OPENAI_API_KEY"],
});
async function main() {
const chatCompletion = await openai.chat.completions.create({
messages: [
{
role: "system",
content: "你是一名 JavaScript 开发者",
},
{
role: "user",
content: "为什么 JavaScript 用于 Web 开发？",
},
],
model: "gpt-4o",
temperature: 0.85,
top_p: 1,
max_tokens: 1921,
frequency_penalty: 0,
presence_penalty: 0,
});
const result = chatCompletion.choices[0].message.content;
console.log(result);
}
main();
```

**清单 3-1** JavaScript ChatGPT 客户端

当你分析清单 3-1 中的代码时，你会看到一些在 Chat Playground 中非常熟悉的内容，例如 `model`、`messages`、`temperature` 和 `tokens`。

## 切勿将你的 API 密钥放入 Web 应用程序！

从清单 3-1 的代码可以看出，只需几行代码就可以在 JavaScript 中创建一个功能完整的 ChatGPT 客户端应用程序。在本章后面，我们将调用该脚本并查看结果。

当然，我们都知道 JavaScript 的主要优势之一是它既可以在服务器端运行，也可以在客户端的浏览器中运行。那么，这是否意味着在你的 React、Angular 或 Vue 项目中使用此代码为网站访问者创造绝佳体验是个好主意呢？

哦，我亲爱的天真孩子。不，不，绝对不行。请记住，Web 应用程序中的所有代码对世界都是 100% 可见的。为了让这段代码在 Web 浏览器中工作，你需要向将在 Web 浏览器中运行的代码提供你的 OpenAI API 密钥。任何有能力的开发者都可以使用大量可用的工具来查看你的 Web 应用程序的源代码并发现你的 OpenAI API 密钥。这意味着他们可以轻松地为你产生巨额账单，并可能以违反 OpenAI 服务条款的方式使用你的 API 密钥。

因此，构建支持 AI 的 Web 应用程序的最佳实践是，在你的服务器端使用 Node.js 进行所有对 OpenAI API 的调用。这样，你的 API 密钥就不可能被暴露。

## 使用 `OpenAI.chat.completions.create()` 向 ChatGPT 发送消息

`OpenAI.chat.completions.create()` 方法基本上是对你在 Chat Playground 中所能做的事情的一对一映射；因此，这个方法应该让你感到得心应手。

表 3-1 描述了调用 `OpenAI.chat.completions.create()` 方法所需参数的格式。虽然表格很长，但快速浏览后，你应该会发现，成功调用该方法实际上只需要几个字段。

该方法的响应称为 `ChatCompletion`。

### 检查方法参数

好的，表 3-1 看起来有点令人生畏！然而，如前所述，只有 `model` 和 `messages` 是必需的参数。

**表 3-1** 创建 ChatCompletion 对象的结构

| 字段 | 类型 | 是否必填 | 描述 |
| --- | --- | --- | --- |
| `model` | `String` | 必填 | 用于`ChatCompletion`的模型 ID。兼容模型包括`gpt-4`、`gpt-4-0613`、`gpt-4-32k`、`gpt-4o`、`gpt-4o-mini`、`o1`、`o1-mini`。 |
| `messages` | `Array` | 必填 | 共有四种消息类型，每种类型都有各自的要求：`系统消息`（见表 3-2）、`用户消息`（见表 3-3）、`助手消息`（见表 3-4）、`工具消息`（见表 3-5）。 |
| `frequency_penalty` | `Number` 或 `null`（默认值：`0`） | 可选 | 介于-2.0 到 2.0 之间的数字。正值会根据词汇在对话历史中出现的频率对其进行惩罚，从而降低逐字重复相同句子的可能性。 |
| `logit_bias` | `JSON Map`（默认值：`null`） | 可选 | 允许你修改特定词汇在补全中出现的可能性。你需要提供一个 JSON 对象，该对象将词汇（由分词器中的词汇 ID 指定）映射到-100 到 100 之间的相关偏差值。此偏差会在采样前添加到模型的 logits 中。 |
| `logprobs` | `boolean` 或 `null` | 可选，默认为`false` | 此参数用于决定是否返回输出词汇的对数概率。当设置为`true`时，它会提供消息内容中包含的每个输出词汇的对数概率。但是，`gpt-4-vision-preview`模型目前不支持此功能。 |
| `max_tokens` | `integer` 或 `null` | 可选 | 此参数设置生成的聊天补全可以拥有的最大词汇数。 |
| `n` | `integer` 或 `null`（默认值：`1`） | 可选 | 指定模型应为每个输入消息生成多少个`ChatCompletion`选项。 |
| `presence_penalty` | `Number` 或 `null`（默认值：`0`） | 可选 | 介于-2.0 到 2.0 之间的数字。正值会根据新词汇是否出现在对话历史中来对其进行惩罚，从而鼓励模型谈论新话题。 |
| `response_format` | `JSON object` | 可选 | 你有两个选项：`{ "type": "json_object" }`用于 JSON 对象响应，或`{ "type": "text" }`用于文本响应。注意：务必记住，在 JSON 模式下操作时，你需要通过系统或用户指令明确命令模型生成 JSON。否则，模型可能会无限输出空白字符，直到达到词汇上限，导致请求看起来像被冻结了一样。此外，请注意，如果`finish_reason`是`"length"`，则表示生成的内容超出了`max_tokens`或对话超过了允许的最大上下文长度，这可能导致消息被截断。 |
| `seed` | `integer` 或 `null` | 可选 | 通过指定种子，系统将尝试生成可重复的结果。理论上，这意味着如果你使用相同的种子和参数重复请求，你应该会收到相同的结果。为了获取用于后续请求的种子值，请从你上一次的响应中复制`system_fingerprint`。 |
| `stop` | `String` / `list` / `null`（默认值：`null`） | 可选 | 你可以提供最多四个序列，API 应在这些序列处停止生成更多词汇。这对于控制响应的长度或内容很有用。 |
| `stream` | `Boolean` 或 `null`（默认值：`false`） | 可选 | 如果`stream`设置为`true`，部分消息更新将作为服务器发送事件发送。这意味着词汇将在可用时作为纯数据事件发送，并且流将以`"data: [DONE]"`消息结束。 |
| `temperature` | `Number` 或 `null`（默认值：`1`） | 可选 | 有效值范围在 0 到 2 之间。控制模型输出的随机性。最佳实践是调整`top_p`或`temperature`，但不要同时调整两者。 |
| `tool_choice` | `String` 或 `JSON object` | 可选 | 此参数控制模型调用哪个（如果有）函数。你有两个选项：`"none"`或`"auto"`。如果你不希望模型调用函数，请使用`"none"`。如果你希望模型在生成消息或调用函数之间进行选择，请使用`"auto"`。通过`{"type": "function", "function": {"name": "my_function"}}`指定特定函数会强制模型调用该函数。请注意，当没有函数时，默认值为`"none"`；当存在函数时，默认值为`"auto"`。 |
| `tools` | `Array` | 可选 | 你可以选择指定模型可能调用的工具列表。目前，仅支持函数作为工具。使用此参数提供模型可能为其生成 JSON 输入的函数列表。 |
| `top_logprobs` | `integer` 或 `null` | 可选 | 可以是 0 到 5 之间的任意整数。用于确定在每个词汇位置返回的最可能词汇的数量，并附带它们各自的对数概率。要使此参数生效，必须通过将`logprobs`设置为`true`来启用它。 |
| `top_p` | `Number` 或 `null`（默认值：`1`） | 可选 | 有效值范围在 0 到 1 之间。指示是考虑少数可能性（0）还是所有可能性（1）。最佳实践是调整`top_p`或`temperature`，但不要同时调整两者。 |
| `user` | `String` | 可选 | 这是一个唯一 ID，你可以选择生成它来代表你的最终用户。这将有助于 OpenAI 监控和检测滥用行为。 |

此外，我们还在上面的代码清单 3-1 中提供了代码，以展示这些参数在实际应用中是如何使用的。所以，正如你所见，作为一名 JavaScript 开发者，我们有几个普通用户无法通过 ChatGPT 网站或聊天游乐场使用的选项和参数。现在，最需要详细解释的参数是`messages`参数，让我们进一步分析它。

## 共有四种消息类型

在以编程方式调用 ChatGPT API 时，你可以向 API 提供四种类型的消息：

- 系统消息
- 用户消息
- 助手消息
- 工具消息

好消息是，如果你回顾第 1 章，我们在那里解释了如何使用聊天游乐场，你会发现我们已经遇到过前三种消息类型！我们目前不熟悉的唯一新消息类型是“工具消息”。

### 系统消息（数组）

**表 3-2** 系统消息的结构

| 字段 | 类型 | 是否必填 | 描述 |
| --- | --- | --- | --- |
| `role` | `String` | 必填 | 必须设置为字符串`"system"`。 |
| `content` | `String` | 必填 | 这些是你希望系统在对话中执行的指令。 |
| `name` | `String` | 可选 | 这是你可以为系统提供的可选名称。 |

代码清单 3-2 是代码清单 3-1 中的一个片段，展示了系统消息的格式：

```
messages=[
{
"role": "system",
"content": "你是一名 JavaScript 开发者"
},
...
```

### 用户消息（数组）

**表 3-3** 用户消息的结构

| 字段 | 类型 | 是否必填 | 描述 |
| --- | --- | --- | --- |
| `role` | `String` | 必填 | 必须设置为字符串`"user"`。 |
| `content` | `String` | 必填 | 此字符串包含你想要发送给 ChatGPT 的实际消息或问题。 |
| `name` | `String` | 可选 | 这是你可以在对话中为自己提供的可选名称。 |

代码清单 3-3 是代码清单 3-1 中的一个片段，展示了用户消息的格式：

```
messages=[
...
{
"role": "user",
"content": "为什么 JavaScript 通常用于数据科学？"
}
...
```

### 助手消息（数组）

**注意**
以防您忘记，助手消息用于“提醒”ChatGPT 它在之前回复中告诉您的内容。理想情况下，这可以让您继续数周或数月前与它的对话。

**表 3-4** 助手消息的结构

| 字段 | 类型 | 是否必需 | 描述 |
| --- | --- | --- | --- |
| `role` | `String` | 必需 | 必须设置为字符串 `"assistant"` |
| `content` | `String` | 必需 | 此字符串包含来自之前对话中 ChatGPT 的回复 |
| `name` | `String` | 可选 | 您可以为对话中的 ChatGPT 提供的可选名称 |
| `tool_calls` | `Array` | 可选 | 如果 ChatGPT 在之前的回复中使用了工具，则在此处包含它指定的工具 |
| `↳ id` | `String` | 必需 | 这是 ChatGPT 调用的工具的 ID |
| `↳ type` | `String` | 必需 | 这是 ChatGPT 调用的工具的类型。只有字面量 `"function"` 是有效的工具类型 |
| `↳ function` | `Object` | 必需 | 这是模型调用的函数 |

清单 3-4 是清单 3-1 中的一个片段，展示了用户消息的格式：

```
messages=
...
{
"role": "assistant",
"content": "JavaScript is typically used for data science for several reasons..."
}
...
```

### 工具消息（数组）

工具消息是一种高级消息类型，用于非常特定的用例。您不能在 ChatGPT 网站或聊天游乐场中使用它们。通过使用工具消息和表 3-1 中的 `tool` 参数，您可以启用 ChatGPT 为您“调用函数”。

乍一看，您可能会想：“哇！ChatGPT 会在云端加载并执行我的代码？太棒了！”不幸的是，事实并非如此。

通过提供函数名称和调用它所需的参数，ChatGPT 会告知您是否应调用该函数以及要放入函数的参数。然后，您需要在您的 JavaScript 代码中**自己**调用该函数。

**表 3-5** 工具消息的结构

| 字段 | 类型 | 是否必需 | 描述 |
| --- | --- | --- | --- |
| `role` | `String` | 必需 | 必须设置为字符串 `"tool"` |
| `content` | `String` | 必需 | 此字符串包含工具消息的内容 |
| `tool_call_id` | `Array` | 可选 | 这是工具调用的 `id` |

**注意**
由于工具消息和函数调用是高级主题，我们不会在本书中进一步解释它们。但是，了解所有四种存在的消息类型是好的。本书将重点介绍系统消息、用户消息和助手消息。

## 运行我们的基本 ChatGPT 客户端

那么，在运行我们在清单 3-1 中创建的代码后，我们可以预期会收到一个类似清单 3-5 中所示的响应。

### 处理响应（`ChatCompletion`）

**表 3-6** `ChatCompletion` 对象响应的结构

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| `id` | `String` | `ChatCompletion` 的唯一标识符 |
| `choices` | `Array` | `ChatCompletion` 选项的列表。如果表 3-1 中的“n”大于 1，则响应中可能有多个选项 |
| `↳ finish_reason` | `String` | 每个响应都会包含一个 `finish_reason`。`finish_reason` 的可能值为：`stop`：API 返回了完整消息，或由通过 `stop` 参数提供的停止序列之一终止的消息 `length`：由于请求中的 `max_tokens` 参数或模型本身的令牌限制，模型输出不完整 `tool_call`：模型调用了工具，例如函数 `content_filter`：由于违反内容过滤器，响应被终止 `null`：API 响应仍在进行中或不完整 |
| `↳ index` | `Integer` | 选项在选项列表中的索引 |
| `↳ message` | `Object` | 模型生成的 `ChatCompletionMessage`。这在表 3-6 中有更详细的解释 |
| `↳ logprobs` | `Object` 或 `Null` | 选项的对数概率信息 |
| `model` | `String` | 用于 `ChatCompletion` 的模型 |
| `system_fingerprint` | `String` | 如果您想从之前的对话中获得可重现的结果，请在后续请求中使用此参数作为“种子” |
| `object` | `String` | 此字段始终返回字面量 `"chat.completion"` |
| `usage` | `Object` | 完成请求的使用统计信息 |
| `↳ completion_tokens` | `Integer` | 生成的完成中的令牌数量 |
| `↳ prompt_tokens` | `Integer` | 提示中的令牌数量 |
| `↳ total_tokens` | `Integer` | 请求中使用的令牌总数，包括提示和完成 |

`ChatCompletion` 对象中最重要的项是 `ChatCompletionMessage`，它在表 3-7 中有更详细的解释。

### `ChatCompletionMessage`

**表 3-7** `ChatCompletionMessage` 的结构

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| `role` | `String` | 该字段的值始终为字面量 `"assistant"` |
| `content` | `String` 或 `null` | 这是一个 `String`，包含 ChatGPT 对我们请求的响应 |
| `tool_calls` | `Array` | 如果你在表 3-1 中指示 ChatGPT 调用某个工具（目前指函数），则该列表将存在于 `ChatCompletionMessage` 中 |
| `↳ id` | `String` | 这是 ChatGPT 所调用工具的 ID |
| `↳ type` | `String` | 这是 ChatGPT 所调用工具的类型。目前只有字面量 `"function"` 是有效的工具类型 |
| `↳ function` | `Object` | 这是模型所调用的函数及其参数 |

## 结论

在本章中，我们结合了第 1 章和第 2 章的经验，用 JavaScript 创建了一个功能完备的 ChatGPT 客户端。在 ChatGPT 客户端的代码中，我们看到了一些在 Chat Playground 中已经介绍过的术语，例如 `model`、`messages`、`temperature` 和 `tokens`。

我们还看到，作为 JavaScript 开发者，OpenAI 为我们提供了**大量额外的选项**来调用 ChatGPT，这些选项是普通日常用户甚至使用 Chat Playground 的技术人员都无法使用的。在本章中，我们花时间解释了这些选项，重点介绍了我们可以发送的 `messages`。

既然我们已经有了一个可运行的 JavaScript ChatGPT 客户端，接下来让我们看看如何在本书的其余示例中利用它！