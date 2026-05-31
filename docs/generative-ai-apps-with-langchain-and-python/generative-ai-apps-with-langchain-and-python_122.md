# 定义提示模板

```python
prompt_template = PromptTemplate(
    input_variables=["topic"],
    template="Generate an engaging article about {topic}."
)
```

### 初始化智能体

```python
agent = initialize_agent(tools, OpenAI(temperature=0), agent="zero-shot-react-description", verbose=True)
```

