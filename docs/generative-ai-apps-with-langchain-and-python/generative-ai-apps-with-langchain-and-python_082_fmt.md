# 第 5 章 掌握创意内容的提示词

## 聊天提示模板

聊天提示模板是提示模板的一种实用变体，在使用大型语言模型处理对话任务时，能让你的工作更加轻松。

正如你之前所学，聊天补全 API 使用消息列表，包括系统消息、人类消息（提示词）和 AI 消息。LangChain 提供了一些有用的类，使处理这些消息变得简单。

### 构建聊天提示模板

让我们看看如何使用`ChatPromptTemplate`类创建这样的模板。

#### 导入所需库

首先，处理基础工作，例如导入如下所示的库：

- `langchain.prompts`：提供`ChatPromptTemplate`、`SystemMessagePromptTemplate`和`HumanMessagePromptTemplate`类，用于创建聊天提示词

- `langchain.chat_models`：提供`ChatOpenAI`类，用于与 OpenAI 聊天模型交互

```python
from langchain.prompts import ChatPromptTemplate,
    SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
import os
```

#### 设置 OpenAI API 密钥

执行设置 API 密钥所需的操作：

```python
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key is None:
    openai_api_key = "your_api_key_here"  # 替换为你的实际 API 密钥
```

#### 创建聊天模型实例

你创建一个名为`chat`的`ChatOpenAI`实例，并指定所需的温度参数（`0`）和 OpenAI API 密钥：

```python
chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
```

#### 定义聊天模板

你使用三引号（`"""`）将聊天模板定义为多行字符串。

- 模板包含一条系统消息，描述助手的角色和行为。

- 它还包含一个用户输入占位符（`{text}`）。

```python
template = """
你是一个充满热情的助手，负责将用户的文本改写得更令人兴奋。

用户：{text}
助手："""
```

#### 创建提示模板

你使用`from_messages()`方法创建一个名为`prompt`的`ChatPromptTemplate`实例。

你向`from_messages()`传递一个消息提示模板列表：

- `SystemMessagePromptTemplate.from_template()`：创建一条系统消息提示模板，包含指定内容

- `HumanMessagePromptTemplate.from_template()`：创建一条人类消息提示模板，包含用户输入的`{text}`占位符

```python
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "你是一个充满热情的助手，负责将用户的文本改写得更令人兴奋。"
    ),
    HumanMessagePromptTemplate.from_template("{text}"),
])
```

#### 获取用户输入