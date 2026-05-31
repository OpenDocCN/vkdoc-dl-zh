# 使用查询运行代理

```python
query = "What is the capital of France? What is the population of that city?"
response = agent.run(query)
print(response)
```

在此示例中，你加载了所需的工具（`serpapi` 和 `llm-math`），并使用这些工具和一个语言模型（`OpenAI`）初始化了零样本 React 代理。然后，你向代理提供一个查询，它会根据查询需求利用适当的工具生成响应。

