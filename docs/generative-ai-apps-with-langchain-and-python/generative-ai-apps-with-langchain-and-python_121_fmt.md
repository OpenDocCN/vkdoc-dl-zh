# 加载必要的工具

```python
tools = load_tools(["serpapi", "llm-math"], llm=OpenAI(temperature=0))
```