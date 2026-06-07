# 初始化你的 LLM 模型

`llm = ChatOpenAI()`

## 创建提示模板

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位世界级的技术文档撰写专家。"),
    ("user", "{input}")
])
```