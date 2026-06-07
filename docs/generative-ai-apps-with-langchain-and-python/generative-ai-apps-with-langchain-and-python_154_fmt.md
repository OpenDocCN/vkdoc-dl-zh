# 设置决策提示

```python
decision_prompt = PromptTemplate(
    input_variables=["data_insights"],
    template="""
基于数据洞察：{data_insights}，
做出应采取何种适当行动的决策。
提供清晰简洁的决策以及简要的理由说明。
"""
)
```

## 初始化智能体

```python
decision_agent = initialize_agent(
    tools,
    OpenAI(temperature=0.7),
    agent="zero-shot-react-description",
    verbose=True
)
```