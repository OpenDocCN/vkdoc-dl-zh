# 加载必要的工具

```
tools = load_tools(["serpapi", "llm-math"], llm=OpenAI(temperature=0))
```

### 初始化智能体

```
agent = initialize_agent(tools, OpenAI(temperature=0), agent="zero-shot-react-description", verbose=True)
```

