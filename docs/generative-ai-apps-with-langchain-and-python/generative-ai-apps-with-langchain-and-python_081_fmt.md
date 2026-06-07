# 第 5 章 掌握创意内容的提示词

## 生成电影信息

- 你定义一个`query`变量，其中包含你想获取信息的电影标题（`"盗梦空间"`）。

- 你创建一个名为`human_message`的`HumanMessage`实例，其内容为使用`prompt.format(query=query)`格式化后的提示词。

- 你将包含`human_message`的列表传递给`llm`实例以生成响应。

```python
# 使用提示词生成电影信息
query = "请介绍电影《盗梦空间》。"
human_message = HumanMessage(content=prompt.format(query=query))
response = llm([human_message])
```

## 解析语言模型的响应

你使用`try-except`块来处理解析过程中可能出现的任何验证错误。

- 你调用`parser.parse(response.content)`，使用初始化的`PydanticOutputParser`解析语言模型响应的内容。

- 如果解析成功，你打印存储在`parsed_movie`中的结构化电影数据。

- 如果解析过程中出现`ValidationError`，你打印错误信息。

```python
# 解析 LLM 的响应
try:
    parsed_movie = parser.parse(response.content)
    # 打印结构化电影数据
    print(parsed_movie)
except ValidationError as e:
    print(f"验证错误：{e}")
```

通过使用`PydanticOutputParser`，你已成功将 LLM 的文本响应转换为结构清晰的电影对象。

此示例展示了输出解析器如何应用于不同领域，例如整理电影、书籍、产品信息，或任何需要从 LLM 响应中提取的结构化数据。

### OutputFixingParser

LangChain 提供了一个便捷工具，名为`OutputFixingParser`。该解析器会获取原始输出，将其重新发送给模型，并要求其修复任何格式问题。这就像拥有一个得力助手，能双重检查模型的工作成果，确保一切准确无误。

我想指出的是，随着模型不断改进，它们遵循指令的能力也越来越强。此外，与 LLM 交互时也存在一定的随机性。因此，如果你在跟随本教程编写代码，可能会发现模型直接生成了正确的输出，一切运行顺畅。这当然是好消息！但即使你无法复现错误也不必担心——你仍然可以从中学习。

在下一节中，你将探索另一种类型的提示词，称为"聊天提示模板"，它非常适合构建聊天机器人和对话代理。