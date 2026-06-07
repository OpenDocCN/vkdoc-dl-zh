# 第 5 章 掌握创意内容的提示词

3. **上下文和问题**：通过提供相关的上下文和具体问题，你可以帮助 LLM 理解对话的背景和焦点。这有助于 LLM 模型提供非常针对你任务的响应。

### 创建多字符串提示模板

让我们看看如何创建和使用适用于任何类型 LLM 的提示模板。首先，你需要从`langchain.prompts`库中导入`PromptTemplate`类。希望你已经导入了 LangChain 库：

```python
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
```

创建一个名为`template`的多行字符串。它类似于 f-string，但使用`{variable}`而不是`{variable}`（开头没有“f”）。

```python
template = """
You are a seasoned software engineer.
Explain the following algorithm: {algorithm} in {language}.
Describe its purpose, time complexity, and a common use case.
"""
```

在这个例子中，你将`{algorithm}`替换为“machine learning”，将`{language}`替换为“French”。生成的提示词将如下所示：你是一位经验丰富的软件工程师。用法语解释以下算法：机器学习。描述其目的、时间复杂度和常见用例。