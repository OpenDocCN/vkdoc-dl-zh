# 第 5 章：掌握创意内容的提示词

## 输出解析器

最后，我们来谈谈输出解析器。有时，你可能希望 LLM 以特定格式生成输出，例如 JSON 或问答模式。你可以使用输出解析器从模型的响应中提取相关信息，并根据你的需求进行结构化处理。

在构建需要从 LLM 输出中提取特定数据类型的应用程序时，输出解析器非常有用。也许你需要一个 Python 的`datetime`对象，或者一个格式良好的 JSON 对象。你可以使用输出解析器，轻松地将 LLM 输出的字符串转换为你所需的确切数据类型，甚至通过`pydantic`转换为自定义的类实例。

输出解析器通常执行两个主要功能：

1.  **获取格式指令**：此方法为 LLM 提供关于信息应如何结构和呈现的指南。
2.  **解析**：一旦 LLM 生成响应，此方法会获取该文本，并根据提供的指令对其进行结构化处理。

当 LLM 的响应与所需格式不完全匹配时，输出解析器提供了一个额外的方法：

1.  **带提示解析**：此方法作为第二次尝试来正确结构化数据。它会考虑 LLM 的响应和原始提示，提供更多上下文来优化输出。

### 输出解析器的类型

LangChain 拥有多样化的输出解析器。无论你需要 JSON 对象，还是处理 CSV 文件，LangChain 很可能都有适合你的解析器。一个很大的优点是，许多解析器支持流式处理，这意味着它们可以处理连续的数据流：

-   **OpenAITools 解析器**：在处理最新的 OpenAI 函数时非常方便。它根据给定的参数、工具和工具选择来结构化输出。
-   **OpenAIFunctions 解析器**：在使用旧的 OpenAI 函数调用参数（如`functions`和`function_call`）时很有用。它很可靠，并以 JSON 对象的形式流式输出。
-   **JSON 解析器**：这是目前最可靠的解析器之一，它返回一个 JSON 对象，如果你希望数据遵循特定模式，可以通过`Pydantic`模型来定义。
-   **XML 解析器**：当你的输出需要是 XML 格式时，使用此解析器。
-   **CSV 解析器**：此解析器将输出转换为整洁的列表，非常适合电子表格。
-   **OutputFixing 解析器**：有时，初稿并不完美。此解析器会包装另一个解析器，并在出现错误时介入，要求 LLM 修复输出。
-   **RetryWithError 解析器**：这与`OutputFixing`类似，但更全面，因为它还会考虑原始输入和指令，并在出现错误时要求 LLM 重做。
-   **Pydantic 解析器**：`Pydantic`帮助你定义模型输出结构，并确保输出完全符合你期望的结构。
-   **YAML 解析器**：用于需要 YAML 格式的输出，当你的数据模式是用 YAML 定义时特别有用。

而这只是冰山一角！还有用于 DataFrame、枚举、日期时间，甚至用于简单任务的基本结构化字典的解析器。关键在于，LangChain 的输出解析器让你能够完全控制如何接收和使用 LLM 生成的数据。

### 示例用例

假设你正在构建一个用于确定科学发现日期的应用程序，并且你希望这些日期是 Python 中的`datetime`对象。你可能会面临两个主要挑战：

1.  LLM 的输出始终是字符串，无论你如何与之交互或提供什么指令。
2.  日期格式可能不同。LLM 可能会回复“January 1st, 2020”、“Jan 1st, 2020”或“2020-01-01”。

这就是解析器可以提供帮助的地方。它们使用格式指令来确保输出格式正确（例如，`datetime`的格式为“2020-01-01”），并提供解析方法将字符串转换为所需的 Python 对象类型。

### 实践示例：使用`PydanticOutputParser`处理电影数据

在此示例中，你将创建一个系统来存储和组织电影信息。

#### 导入所需库

你需要导入以下库：

-   `pydantic`：提供`BaseModel`、`Field`、`ValidationError`和`field_validator`类，用于定义数据模型和验证
-   `langchain.output_parsers`：提供`PydanticOutputParser`类，用于解析语言模型的输出
-   `langchain.prompts`：提供`PromptTemplate`类，用于创建提示模板
-   `langchain.chat_models`：提供`ChatOpenAI`类，用于与 OpenAI 聊天模型交互
-   `langchain.schema`：提供`HumanMessage`类，用于表示对话中的人类消息

```python
from pydantic import BaseModel, Field, ValidationError, field_validator
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
```

#### 定义电影数据模型

在这里，你定义了电影数据模型：

-   你定义了一个继承自`BaseModel`的`Movie`类，用于表示电影的结构。
-   该类有三个字段：`title`、`director`和`year`，每个字段都有描述。
-   你使用`@field_validator`装饰器来确保电影标题是首字母大写的。如果标题不是标题大小写格式，则会引发`ValidationError`。

```python
class Movie(BaseModel):
    title: str = Field(description="电影标题")
    director: str = Field(description="电影导演")
    year: int = Field(description="电影发行年份")

    # 质量控制：确保电影标题首字母大写
    @field_validator('title')
    def title_must_be_capitalized(cls, value):
        if not value.istitle():
            raise ValueError("电影标题必须首字母大写。")
        return value
```

#### 初始化输出解析器

你创建一个名为`parser`的`PydanticOutputParser`实例，并将`Movie`类作为`pydantic_object`参数传递，以指定所需的输出结构：

```python
# 使用我们的数据结构初始化解析器
parser = PydanticOutputParser(pydantic_object=Movie)
```

#### 创建提示模板

你定义一个名为`prompt`的`PromptTemplate`，它向语言模型提供关于如何格式化电影信息的指令：

-   模板包含格式指令（`{format_instructions}`）和查询（`{query}`）的占位符。
-   你指定输入变量`query`，并提供从解析器通过`parser.get_format_instructions()`获取的格式指令。



